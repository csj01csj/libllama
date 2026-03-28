package com.aicode.studio.ai

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.*
import android.util.Log
import com.aicode.studio.engine.AIInferenceService
import com.aicode.studio.engine.InferenceConfig
import com.aicode.studio.engine.ModelSelectActivity
import com.aicode.studio.util.LogManager
import org.json.JSONArray
import org.json.JSONObject

/**
 * com.aicode.studio.engine.AIInferenceService와 IPC 통신하는 클라이언트.
 * - bindService로 Messenger 연결 (로컬 서비스)
 * - sendPrompt로 프롬프트 전송, 토큰 스트리밍 수신
 * - stopAndDisconnect로 서비스 종료 (앱 종료 / 스위치 OFF 시 호출)
 */
class LocalAIManager(private val context: Context, private val logger: LogManager) {

    companion object {
        private const val TAG = "LocalAIManager"
    }

    interface StreamCallback {
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    private var engineMessenger : Messenger? = null
    private var replyMessenger  : Messenger? = null
    private var streamCallback  : StreamCallback? = null
    private val responseBuffer  = StringBuilder()
    private var _connected      = false

    // ── 수신 핸들러 ────────────────────────────────────────────
    private val replyHandler = object : Handler(Looper.getMainLooper()) {
        override fun handleMessage(msg: Message) {
            when (msg.what) {
                InferenceConfig.MSG_TOKEN_STREAM -> {
                    val token = msg.data.getString(InferenceConfig.KEY_TOKEN) ?: ""
                    responseBuffer.append(token)
                    streamCallback?.onToken(token)
                }
                InferenceConfig.MSG_GEN_COMPLETE -> {
                    val full = responseBuffer.toString()
                    responseBuffer.clear()
                    streamCallback?.onComplete(full)
                }
                InferenceConfig.MSG_STATUS_REPLY -> {
                    val info = msg.data.getString("info") ?: ""
                    if (info.startsWith("ERROR:")) {
                        responseBuffer.clear()
                        streamCallback?.onError(info.removePrefix("ERROR:"))
                    } else if (info.startsWith("READY:")) {
                        logger.logSystem("Local AI 준비됨: ${info.removePrefix("READY:")}")
                    }
                }
            }
        }
    }

    // ── 서비스 연결 ────────────────────────────────────────────
    private val conn = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            engineMessenger = Messenger(binder)
            _connected = true
            logger.logSystem("Local AI 엔진 연결됨")
        }
        override fun onServiceDisconnected(name: ComponentName?) {
            engineMessenger = null
            _connected = false
            logger.logSystem("Local AI 엔진 연결 끊김")
        }
    }

    fun connect() {
        if (_connected) return
        replyMessenger = Messenger(replyHandler)
        val intent = Intent(context, AIInferenceService::class.java)
        try {
            context.bindService(intent, conn, Context.BIND_AUTO_CREATE)
        } catch (e: Exception) {
            logger.logError("Local AI 연결 실패: ${e.message}")
        }
    }

    /** 서비스 바인드만 해제 (서비스는 계속 실행 — 액티비티 종료 시 사용) */
    fun disconnect() {
        try { context.unbindService(conn) } catch (_: Exception) {}
        engineMessenger = null
        _connected = false
    }

    /** 서비스 바인드 해제 + 서비스 종료 (스위치 OFF 시 사용) */
    fun stopAndDisconnect() {
        try {
            val stopIntent = Intent(context, AIInferenceService::class.java).apply {
                action = AIInferenceService.ACTION_STOP
            }
            context.startService(stopIntent)
        } catch (_: Exception) {}
        try { context.unbindService(conn) } catch (_: Exception) {}
        engineMessenger = null
        _connected = false
    }

    fun isConnected() = _connected && engineMessenger != null

    fun setStreamCallback(cb: StreamCallback) { streamCallback = cb }
    fun clearStreamCallback() { streamCallback = null }

    // ── 컨텍스트(시스템 프롬프트) 주입 ──────────────────────────
    fun setContext(contextStr: String) {
        val msg = Message.obtain(null, InferenceConfig.MSG_SET_CONTEXT).apply {
            data = Bundle().apply { putString(InferenceConfig.KEY_CONTEXT_JSON, contextStr) }
        }
        try { engineMessenger?.send(msg) } catch (e: Exception) {
            Log.e(TAG, "컨텍스트 전송 실패", e)
        }
    }

    // ── 프롬프트 전송 ─────────────────────────────────────────
    fun sendPrompt(prompt: String) {
        if (!isConnected()) { streamCallback?.onError("Local AI 미연결"); return }
        responseBuffer.clear()
        val msg = Message.obtain(null, InferenceConfig.MSG_SEND_PROMPT).apply {
            data = Bundle().apply { putString(InferenceConfig.KEY_PROMPT, prompt) }
            replyTo = replyMessenger
        }
        try { engineMessenger?.send(msg) } catch (e: Exception) {
            streamCallback?.onError("전송 실패: ${e.message}")
        }
    }

    // ── 대화 기록 주입 (매 요청 전 AIAgentManager가 호출) ─────
    fun importHistory(history: List<Pair<String, String>>) {
        val arr = JSONArray()
        history.forEach { (u, a) ->
            arr.put(JSONObject().apply { put("u", u); put("a", a) })
        }
        val msg = Message.obtain(null, InferenceConfig.MSG_SET_HISTORY).apply {
            data = Bundle().apply { putString(InferenceConfig.KEY_HISTORY, arr.toString()) }
        }
        try { engineMessenger?.send(msg) } catch (_: Exception) {}
    }

    // ── 대화 기록 초기화 ──────────────────────────────────────
    fun clearHistory() {
        val msg = Message.obtain(null, InferenceConfig.MSG_CLEAR_HISTORY)
        try { engineMessenger?.send(msg) } catch (_: Exception) {}
    }

    // ── 생성 중단 ─────────────────────────────────────────────
    fun stopGeneration() {
        try { engineMessenger?.send(Message.obtain(null, InferenceConfig.MSG_STOP_GEN)) } catch (_: Exception) {}
    }

    /** 모델 관리 화면 열기 (로컬 ModelSelectActivity) */
    fun openManageActivity() {
        try {
            val intent = Intent(context, ModelSelectActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            context.startActivity(intent)
        } catch (e: Exception) {
            logger.logError("Local AI 관리 화면 열기 실패: ${e.message}")
        }
    }
}
