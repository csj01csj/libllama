package com.aicode.studio.engine

import android.app.*
import android.content.Intent
import android.os.*
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import android.os.Build

/**
 * 백그라운드 AI 추론 포그라운드 서비스 (직접 llama.cpp JNI 기반).
 *
 * llamatik의 LlamaBridge 대신 LlamaBridgeDirect를 사용합니다.
 * n_ctx / n_batch / n_ubatch를 직접 제어하여 SIGABRT 크래시를 방지합니다.
 */
class AIInferenceService : Service() {

    companion object {
        private const val TAG = "AIInferenceService"
        const val ACTION_START     = "com.aicode.studio.engine.START"
        const val ACTION_STOP      = "com.aicode.studio.engine.STOP"
        const val BROADCAST_STATUS = "com.aicode.studio.engine.STATUS"
        const val EXTRA_STATUS     = "status"
    }

    enum class State { IDLE, LOADING, READY, GENERATING, ERROR }

    private var state           : State = State.IDLE
    private var activeModel     : InferenceConfig.ModelDef? = null
    private var pendingModel    : InferenceConfig.ModelDef? = null   // set at ACTION_START, loaded on first generate
    private var modelLoaded     : Boolean = false
    private var thinkingEnabled : Boolean = true
    private var pendingCtxJson  : String  = ""

    // ── 대화 기록 (nUbatch=512 제한으로 최대 3턴 / 600자 이내 유지) ──
    private val localHistory   = mutableListOf<Pair<String, String>>()
    private var lastUserInput  = ""
    private val genBuffer      = StringBuilder()

    private val inferScope     = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var genJob         : Job? = null
    private val clients        = mutableSetOf<Messenger>()
    private val inferenceMutex = Mutex()

    /** Single-thread executor for all native llama.cpp calls */
    private val nativeExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private val nativeContext  = nativeExecutor.asCoroutineDispatcher()

    private val handler = object : Handler(Looper.getMainLooper()) {
        override fun handleMessage(msg: Message) {
            msg.replyTo?.let { clients.add(it) }
            when (msg.what) {
                InferenceConfig.MSG_SEND_PROMPT -> {
                    val prompt = msg.data.getString(InferenceConfig.KEY_PROMPT) ?: return
                    doGenerate(prompt)
                }
                InferenceConfig.MSG_STOP_GEN -> {
                    LlamaBridgeDirect.nativeStopGeneration()  // thread-safe atomic flag
                    genJob?.cancel()
                    genJob = null
                    setState(State.READY)
                }
                InferenceConfig.MSG_GET_STATUS -> {
                    val profile = HardwareAnalyzer.analyze(applicationContext)
                    msg.replyTo?.send(Message.obtain(null, InferenceConfig.MSG_STATUS_REPLY).apply {
                        data = Bundle().also {
                            it.putString("state",              state.name)
                            it.putString("model_name",         activeModel?.displayName ?: "")
                            it.putString("backend",            profile.recommendedBackend.displayName)
                            it.putString("hw_info",            "${profile.gpuRenderer} / ${if (profile.hasDotProd) "DotProd" else "NEON"}")
                            it.putBoolean("thinking",          thinkingEnabled)
                            it.putBoolean("supports_thinking", activeModel?.supportsThinking ?: false)
                        }
                    })
                }
                InferenceConfig.MSG_SET_CONTEXT -> {
                    pendingCtxJson = msg.data.getString(InferenceConfig.KEY_CONTEXT_JSON) ?: ""
                }
                InferenceConfig.MSG_SET_THINKING -> {
                    thinkingEnabled = msg.data.getBoolean(InferenceConfig.KEY_THINKING, true)
                }
                InferenceConfig.MSG_CLEAR_HISTORY -> {
                    localHistory.clear()
                }
                InferenceConfig.MSG_SET_HISTORY -> {
                    val json = msg.data.getString(InferenceConfig.KEY_HISTORY) ?: return
                    localHistory.clear()
                    try {
                        val arr = JSONArray(json)
                        for (i in 0 until arr.length()) {
                            val o = arr.getJSONObject(i)
                            localHistory.add(o.getString("u") to o.getString("a"))
                        }
                    } catch (_: Exception) {}
                }
            }
        }
    }

    private val messenger = Messenger(handler)

    override fun onCreate() {
        super.onCreate()
        startForeground(InferenceConfig.NOTIF_ID, buildNotif("AI 엔진 대기 중"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val ctxJson = intent?.getStringExtra(InferenceConfig.KEY_CONTEXT_JSON)
        if (ctxJson != null) pendingCtxJson = ctxJson

        when (intent?.action) {
            ACTION_START -> {
                val id    = intent.getStringExtra(InferenceConfig.KEY_MODEL_ID) ?: return START_STICKY
                val model = InferenceConfig.ALL_MODELS.firstOrNull { it.id == id } ?: return START_STICKY
                // 모델이 변경되면 클라이언트에 알림
                val prevModel = activeModel
                if (prevModel != null && prevModel.id != model.id) {
                    broadcastRaw("MODEL_CHANGED:${prevModel.displayName} → ${model.displayName}")
                }
                // Store the model to load, but do NOT load it yet.
                // Model loading starts lazily on the first doGenerate() call.
                // This prevents the heavy model load (and potential GPU crash) at app startup.
                pendingModel = model
                activeModel  = model
                setState(State.READY)
            }
            ACTION_STOP -> stopSelf()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder  = messenger.binder
    override fun onUnbind(intent: Intent?): Boolean = true

    override fun onDestroy() {
        super.onDestroy()
        LlamaBridgeDirect.nativeStopGeneration()
        inferScope.cancel()
        // Schedule shutdown on the native thread (runs after any pending generation)
        nativeExecutor.execute {
            runCatching { LlamaBridgeDirect.nativeShutdown() }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Model loading
    // ─────────────────────────────────────────────────────────────────────────

    private fun loadModel(model: InferenceConfig.ModelDef, afterLoad: (() -> Unit)? = null) {
        if (state == State.LOADING) return
        setState(State.LOADING)
        updateNotif("${model.displayName} 로드 중…")

        inferScope.launch {
            try {
                val mm      = ModelManager(applicationContext)
                val file    = mm.modelFile(model)
                require(file.exists() && file.length() > 0) { "모델 파일 없음: ${file.absolutePath}" }

                val profile = HardwareAnalyzer.analyze(applicationContext)
                val ram     = profile.totalRamGb

                val nCtx    = when {
                    ram >= 16 -> 4096
                    ram >= 12 -> 2048
                    ram >= 8  -> 1536
                    else      -> 1024
                }
                val nBatch  = nCtx
                val nUbatch = nBatch.coerceAtMost(512)

                // GPU layers: attempt Vulkan when the hardware profile recommends it.
                // The JNI layer (safe_backend_init) will catch any driver crash via
                // sigsetjmp/siglongjmp and automatically fall back to CPU-only,
                // so it is safe to pass n_gpu_layers > 0 here unconditionally.
                val nGpuLayers = if (profile.hasVulkan) 99 else 0
                val nThreads   = profile.cores.coerceAtMost(8)

                val ret = withContext(nativeContext) {
                    if (modelLoaded) {
                        runCatching { LlamaBridgeDirect.nativeShutdown() }
                        modelLoaded = false
                    }
                    LlamaBridgeDirect.nativeInit(
                        modelPath   = file.absolutePath,
                        nCtx        = nCtx,
                        nBatch      = nBatch,
                        nUbatch     = nUbatch,
                        nGpuLayers  = nGpuLayers,
                        nThreads    = nThreads
                    )
                }

                if (ret != 0) {
                    throw RuntimeException("nativeInit 실패 (code $ret)")
                }

                modelLoaded = true
                activeModel = model
                if (!model.supportsThinking) thinkingEnabled = false

                val gpuOk = LlamaBridgeDirect.nativeIsGpuAvailable()
                val backendLabel = if (gpuOk) "GPU (Vulkan)" else "CPU"
                setState(State.READY)
                updateNotif("${model.displayName}${if (model.supportsThinking && thinkingEnabled) " 🧠" else ""} · $backendLabel")
                broadcastRaw("READY:${model.displayName}|$backendLabel")
                Log.d(TAG, "모델 로드 완료: ${model.displayName} | $backendLabel | ctx=$nCtx batch=$nBatch ubatch=$nUbatch threads=$nThreads")

                afterLoad?.invoke()

            } catch (e: Exception) {
                Log.e(TAG, "모델 로드 실패", e)
                setState(State.ERROR)
                updateNotif("로드 실패: ${e.message?.take(50)}")
                broadcastRaw("ERROR:로드 실패 - ${e.message}")
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Prompt helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Qwen2.5 / Qwen3 ChatML 포맷 (대화 기록 포함).
     * parse_special=true로 tokenise하므로 <|im_start|>/<|im_end|>가 단일 특수 토큰으로 처리됩니다.
     */
    private fun buildQwenPrompt(systemPrompt: String, history: List<Pair<String, String>>, userPrompt: String): String = buildString {
        append("<|im_start|>system\n")
        append(systemPrompt.trim())
        append("\n<|im_end|>\n")
        for ((hu, ha) in history) {
            append("<|im_start|>user\n${hu.trim()}\n<|im_end|>\n")
            append("<|im_start|>assistant\n${ha.trim()}\n<|im_end|>\n")
        }
        append("<|im_start|>user\n")
        append(userPrompt.trim())
        append("\n<|im_end|>\n")
        append("<|im_start|>assistant\n")
    }

    /** 최근 대화 기록 — 총 600자, 최대 3턴으로 제한 (nUbatch=512 토큰 예산 보호) */
    private fun recentHistory(): List<Pair<String, String>> {
        var total = 0
        val result = mutableListOf<Pair<String, String>>()
        for ((u, a) in localHistory.reversed()) {
            val size = u.length + a.length + 50 // 태그 오버헤드
            if (total + size > 600) break
            result.add(0, u to a)
            total += size
        }
        return result
    }

    /** 생성된 응답을 기록에 추가 (thinking 블록 제거, 각 메시지 200자로 절삭) */
    private fun addToHistory(user: String, assistant: String) {
        val clean = assistant.replace(Regex("<think>[\\s\\S]*?</think>\\s*"), "").trim()
        if (clean.isEmpty()) return
        localHistory.add(user.take(200) to clean.take(200))
        if (localHistory.size > 3) localHistory.removeAt(0)
    }

    private fun buildSystemPrompt(ctxJson: String): String = buildString {
        appendLine("You are an expert AI coding assistant embedded in Aicode Studio, a mobile Android IDE.")
        appendLine("Respond in the same language as the user (Korean if they write Korean).")
        appendLine()
        appendLine("## File Operations")
        appendLine("You can operate on project files by outputting a JSON block (omit unused keys):")
        appendLine("""{"thought":"plan...","updates":[{"path":"rel/path","content":"full content"}],"patch":{"path":"file","old":"exact string","new":"replacement"},"read_range":{"path":"file","start":1,"end":50},"grep":{"path":".","pattern":"regex","recursive":true},"deletes":["path"]}""")
        appendLine("Rules: paths are relative to project root. 'patch' old= must be exact match incl. whitespace.")
        if (ctxJson.isNotBlank()) {
            appendLine()
            appendLine("## Project Context")
            appendLine(ctxJson)
        }
    }.trim()

    // ─────────────────────────────────────────────────────────────────────────
    //  Generation
    // ─────────────────────────────────────────────────────────────────────────

    private fun doGenerate(userInput: String) {
        // Lazy model load: if a model is pending but not yet loaded, load it now
        if (!modelLoaded) {
            val toLoad = pendingModel ?: run { broadcastRaw("ERROR:모델 미선택"); return }
            pendingModel = null
            loadModel(toLoad, afterLoad = { doGenerate(userInput) })
            return
        }
        genJob?.cancel()
        setState(State.GENERATING)

        genJob = inferScope.launch {
            inferenceMutex.withLock {
                try {
                    delay(300)
                    yield()

                    val model      = activeModel!!
                    val sysPrompt  = buildSystemPrompt(pendingCtxJson)
                    val userMsg    = if (model.supportsThinking) {
                        "$userInput ${if (thinkingEnabled) "/think" else "/no_think"}"
                    } else userInput

                    lastUserInput = userInput
                    genBuffer.clear()

                    val prompt = buildQwenPrompt(sysPrompt, recentHistory(), userMsg)
                    Log.d(TAG, "프롬프트 길이: ${prompt.length}자 (기록 ${localHistory.size}턴)")
                    updateNotif("${model.displayName} 생성 중…")

                    withContext(nativeContext) {
                        LlamaBridgeDirect.nativeGenerate(
                            prompt      = prompt,
                            temperature = 0.7f,
                            maxTokens   = 512,
                            topK        = 40,
                            topP        = 0.95f,
                            callback    = object : LlamaBridgeDirect.TokenCallback {
                                override fun onToken(piece: String) {
                                    genBuffer.append(piece)
                                    val m = Message.obtain(null, InferenceConfig.MSG_TOKEN_STREAM)
                                    m.data = Bundle().apply { putString(InferenceConfig.KEY_TOKEN, piece) }
                                    sendToAll(m)
                                }
                                override fun onComplete() {
                                    sendToAll(Message.obtain(null, InferenceConfig.MSG_GEN_COMPLETE))
                                    setState(State.READY)
                                    updateNotif("${model.displayName} 대기 중")
                                }
                                override fun onError(error: String) {
                                    Log.e(TAG, "네이티브 엔진 에러: $error")
                                    broadcastRaw("ERROR:$error")
                                    setState(State.READY)
                                    updateNotif("오류: $error")
                                }
                            }
                        )
                    }

                } catch (e: CancellationException) {
                    LlamaBridgeDirect.nativeStopGeneration()
                    setState(State.READY)
                } catch (e: Exception) {
                    Log.e(TAG, "생성 예외", e)
                    broadcastRaw("ERROR:${e.message}")
                    setState(State.READY)
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  IPC helpers
    // ─────────────────────────────────────────────────────────────────────────

    private fun sendToAll(msg: Message) {
        val dead = mutableSetOf<Messenger>()
        for (c in clients) {
            try { c.send(Message.obtain(msg)) } catch (_: Exception) { dead.add(c) }
        }
        clients.removeAll(dead)
    }

    private fun broadcastRaw(info: String) {
        sendToAll(Message.obtain(null, InferenceConfig.MSG_STATUS_REPLY).apply {
            data = Bundle().also { it.putString("info", info) }
        })
    }

    private fun setState(s: State) {
        state = s
        // 생성/로드 중에만 포그라운드 유지; 대기/유휴 시엔 포그라운드 해제
        when (s) {
            State.GENERATING, State.LOADING -> {
                val text = when (s) {
                    State.LOADING    -> "${activeModel?.displayName ?: "AI"} 로드 중…"
                    else             -> "${activeModel?.displayName ?: "AI"} 생성 중…"
                }
                startForeground(InferenceConfig.NOTIF_ID, buildNotif(text))
            }
            else -> {
                @Suppress("DEPRECATION")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                } else {
                    stopForeground(true)
                }
            }
        }
        sendBroadcast(Intent(BROADCAST_STATUS).apply {
            putExtra(EXTRA_STATUS, s.name)
            setPackage(packageName)
        })
    }

    private fun buildNotif(text: String): Notification {
        val openIntent = packageManager.getLaunchIntentForPackage(packageName)
        val openPi = PendingIntent.getActivity(this, 0, openIntent, PendingIntent.FLAG_IMMUTABLE)
        val stopPi = PendingIntent.getService(this, 0,
            Intent(this, AIInferenceService::class.java).apply { action = ACTION_STOP },
            PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, InferenceConfig.NOTIF_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("AICode Engine")
            .setContentText(text)
            .setContentIntent(openPi)
            .addAction(android.R.drawable.ic_media_pause, "중지", stopPi)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotif(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(InferenceConfig.NOTIF_ID, buildNotif(text))
    }
}
