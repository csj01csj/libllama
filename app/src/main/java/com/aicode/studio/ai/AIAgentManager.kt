package com.aicode.studio.ai

import com.aicode.studio.util.LogManager
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * AI 에이전트 매니저
 * - Cloud AI (Gemini API) 또는 Local AI (com.aicode.engine) 라우팅
 * - MemoryManager로 기억 파일 관리 (트리밍 + 아카이브 + 요약 인덱스)
 */
class AIAgentManager(private val logger: LogManager) {

    data class ApiConfig(val key: String, val model: String)

    interface Callback {
        fun onThought(thought: String)
        fun onFileUpdated(path: String, success: Boolean, msg: String)
        fun onDeleteRequested(path: String, onConfirm: (Boolean) -> Unit)
        fun onCompleted(summary: String)
        fun onError(error: String)
        fun onStatusChanged(status: String)
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val prompt = PromptBuilder()
    private var history = JSONArray()
    private var root: File? = null
    private var cb: Callback? = null
    private var running = false

    // Cloud AI
    private var apiConfigs = mutableListOf<ApiConfig>()
    private var currentApiIdx = 0

    // Local AI
    private var localAI: LocalAIManager? = null
    private var localMode = false
    private var localModelName = "Local AI"

    // Memory
    private var memoryMgr: MemoryManager? = null
    private var archiveSummary: MutableList<MemoryManager.ArchiveEntry> = mutableListOf()
    private val localConvHistory = mutableListOf<Pair<String, String>>()

    private var iter = 0
    private val MAX_ITER = 15

    // ── 설정 ──────────────────────────────────────────────────

    fun setCallback(c: Callback) { cb = c }

    fun setApiConfigs(configs: List<ApiConfig>) {
        apiConfigs = configs.toMutableList()
        currentApiIdx = 0
    }

    fun setLocalAI(manager: LocalAIManager?, enabled: Boolean) {
        localAI = manager
        localMode = enabled
    }

    fun setLocalModelName(name: String) { localModelName = name }

    fun isRunning() = running

    // ── 프로젝트 로드 ──────────────────────────────────────────

    fun loadProject(r: File) {
        root = r
        memoryMgr = MemoryManager(r)
        loadSession()
    }

    // ── 실행 ──────────────────────────────────────────────────

    fun execute(command: String) {
        val r = root ?: run { cb?.onError("프로젝트 미선택"); return }

        if (localMode) {
            if (localAI?.isConnected() != true) { cb?.onError("Local AI 미연결 - 스위치를 켜고 엔진이 준비될 때까지 기다려주세요"); return }
        } else {
            if (apiConfigs.isEmpty()) { cb?.onError("API 키가 설정되지 않았습니다."); return }
        }
        if (running) { cb?.onError("이미 실행 중"); return }

        running = true; iter = 0; currentApiIdx = 0
        cb?.onStatusChanged(if (localMode) "Local AI 분석 중..." else "AI 분석 중 (API 1)...")

        // 백그라운드: 메모리 트리밍 → 프롬프트 빌드 → 전송
        Thread {
            // ── 로컬 AI: 경량 에이전트 루프 ────────────────────────────────────
            // Qwen3/Qwen2.5 모델은 non-causal (hybrid chunked) attention 사용.
            // llamatik 기본 n_ubatch=512 이므로 프롬프트 토큰 수가 512를 초과하면
            // "noncausal attention requires n_ubatch >= n_tokens" assertion → ggml_abort 발생.
            // PromptBuilder의 에이전트 시스템 프롬프트는 700-1000+ 토큰이므로 반드시 우회 필요.
            if (localMode) {
                val local = localAI ?: run { running = false; cb?.onError("Local AI 없음"); return@Thread }
                // 저장된 대화 기록 주입 (서비스는 매번 덮어씀)
                local.importHistory(recentLocalHistory())
                executeLocalAI(local, command, r, 0)
                return@Thread
            }

            // ── Cloud AI: 전체 에이전트 프레임워크 ─────────────────────────────
            // 1. 메모리 트리밍 체크
            val mgr = memoryMgr
            if (mgr != null && mgr.needsTrimming(MemoryManager.Session(archiveSummary, history))) {
                logger.logSystem("기억 파일 용량 초과 - 정리 중...")
                val summarizer: ((String) -> String?)? = if (apiConfigs.isNotEmpty()) {
                    { text -> summarizeBlocking(text) }
                } else null
                val trimmed = mgr.trimAndArchive(MemoryManager.Session(archiveSummary, history), summarizer)
                history = trimmed.messages
                archiveSummary = trimmed.archiveSummary
                mgr.save(trimmed)
                logger.logSystem("기억 정리 완료 (아카이브 ${archiveSummary.size}개)")
            }

            // 2. 아카이브 인덱스 컨텍스트
            val archiveCtx = mgr?.buildArchiveIndexText(MemoryManager.Session(archiveSummary, history)) ?: ""

            // 3. 프롬프트 빌드
            val localPrompt = PromptBuilder(60000)

            val sys = localPrompt.buildSystemPrompt(r, currentApiIdx + 1, archiveCtx)
            val usr = localPrompt.buildUserPrompt(command, r)
            val msg = JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray().put(JSONObject().put("text", "$sys\n\n$usr")))
            }
            history.put(msg)

            // 4. 전송
            sendNext()
        }.start()
    }

    fun resetSession() {
        history = JSONArray()
        archiveSummary = mutableListOf()
        localConvHistory.clear()
        iter = 0; running = false
        localAI?.clearHistory()
        root?.let { saveSession() }
    }

    fun stop() {
        running = false
        if (localMode) localAI?.stopGeneration()
        saveSession()
        cb?.onStatusChanged("중지됨")
    }

    // ── Cloud AI 통신 ──────────────────────────────────────────

    private fun sendNext() {
        if (!running) return
        if (currentApiIdx >= apiConfigs.size) {
            running = false; cb?.onError("사용 가능한 API 없음 (모두 소진됨)"); return
        }

        val config = apiConfigs[currentApiIdx]
        iter++
        if (iter > MAX_ITER) {
            running = false; cb?.onError("최대 반복(${MAX_ITER})회 도달, 작업 중단")
            saveSession(); return
        }
        logger.logAI("API ${currentApiIdx + 1} (${config.model}) - Iteration $iter")

        val body = JSONObject().apply {
            put("contents", history)
            put("generationConfig", JSONObject().apply {
                put("temperature", 0.2); put("maxOutputTokens", 8192)
            })
        }.toString().toRequestBody("application/json".toMediaType())

        val url = "https://generativelanguage.googleapis.com/v1beta/models/${config.model}:generateContent?key=${config.key}"
        val req = Request.Builder().url(url).post(body).build()

        client.newCall(req).enqueue(object : okhttp3.Callback {
            override fun onResponse(call: Call, resp: Response) {
                val code = resp.code
                val raw  = resp.body?.string() ?: ""
                if (code != 200) {
                    logger.logError("API ${currentApiIdx + 1} (${config.model}): err code:$code")
                    if (code == 429 || raw.contains("quota", true) || raw.contains("exhausted", true)) {
                        currentApiIdx++
                        mainHandler { cb?.onStatusChanged("API $currentApiIdx 소진됨. 다음 시도..."); sendNext() }
                    } else {
                        running = false; cb?.onError("err code:$code")
                    }
                    return
                }
                try { handleResp(raw, config.model, currentApiIdx + 1) }
                catch (e: Exception) {
                    logger.logError("Parse error: ${e.message}")
                    running = false; cb?.onError("응답 파싱 실패")
                }
            }
            override fun onFailure(call: Call, e: IOException) {
                logger.logError("Network API ${currentApiIdx + 1}: ${e.message}")
                currentApiIdx++
                mainHandler { sendNext() }
            }
        })
    }

    // ── 응답 처리 ──────────────────────────────────────────────

    private fun handleResp(raw: String, modelName: String, apiIdx: Int) {
        val originalText = if (modelName == "Local AI") {
            raw  // Local AI는 이미 텍스트
        } else {
            val rj    = JSONObject(raw)
            val cands = rj.optJSONArray("candidates")
            if (cands == null || cands.length() == 0) {
                running = false; saveSession(); cb?.onError("빈 응답"); return
            }
            val aiContent = cands.getJSONObject(0).getJSONObject("content")
            val text      = aiContent.getJSONArray("parts").getJSONObject(0).getString("text")
            // 히스토리에 AI 응답 추가 (provenance 포함)
            aiContent.getJSONArray("parts").getJSONObject(0)
                .put("text", "[Result from $modelName (API $apiIdx)]\n$text")
            history.put(aiContent)
            saveSession()
            text
        }

        // Local AI 응답은 직접 히스토리에 추가
        if (modelName == "Local AI") {
            history.put(JSONObject().apply {
                put("role", "model")
                put("parts", JSONArray().put(JSONObject().put("text",
                    "[Result from Local AI: $localModelName]\n$originalText")))
            })
            saveSession()
        }

        val json = extractJson(originalText)
        if (json == null) {
            running = false; saveSession()
            // 너무 긴 원문은 로그에서 잘라서 표시
            val display = if (originalText.length > 500) originalText.take(500) + "\n...(truncated)" else originalText
            cb?.onCompleted(display)
            return
        }

        val result  = JSONObject(json)
        val thought = result.optString("thought", "")
        if (thought.isNotEmpty()) cb?.onThought(thought)

        val feedback = mutableListOf<String>()
        processTools(result, feedback)
    }

    // ── 도구 처리 ──────────────────────────────────────────────

    private fun processTools(result: JSONObject, feedback: MutableList<String>) {
        val grep      = result.optJSONObject("grep")
        val readRange = result.optJSONObject("read_range")
        val patch     = result.optJSONObject("patch")
        val updates   = result.optJSONArray("updates")
        val deletes   = result.optJSONArray("deletes")

        var actionTaken = false
        if (grep != null)      { actionTaken = true; feedback.add(executeGrep(grep.optString("path","."), grep.optString("pattern",""), grep.optBoolean("recursive",true))) }
        if (readRange != null) { actionTaken = true; feedback.add(executeReadRange(readRange.optString("path"), readRange.optInt("start",1), readRange.optInt("end",100))) }
        if (patch != null)     { actionTaken = true; feedback.add(executePatch(patch.optString("path"), patch.optString("old"), patch.optString("new"))) }

        if (updates != null && updates.length() > 0) {
            for (i in 0 until updates.length()) {
                val o = updates.getJSONObject(i)
                val p = o.getString("path"); val c = o.getString("content")
                try {
                    val f = File(root, p); f.parentFile?.mkdirs(); f.writeText(c)
                    feedback.add("OK: Wrote $p"); cb?.onFileUpdated(p, true, "완료")
                } catch (e: Exception) { feedback.add("FAIL: $p - ${e.message}") }
            }
        }

        if (deletes != null && deletes.length() > 0) {
            processDeletes(deletes, 0, feedback)
        } else {
            if (!actionTaken && (updates == null || updates.length() == 0)) {
                running = false; saveSession(); cb?.onCompleted("작업 완료"); return
            }
            sendFeedback(feedback)
        }
    }

    private fun processDeletes(del: JSONArray, idx: Int, fb: MutableList<String>) {
        if (idx >= del.length()) { sendFeedback(fb); return }
        val p = del.getString(idx)
        cb?.onDeleteRequested(p) { ok ->
            if (ok) { File(root, p).deleteRecursively(); fb.add("OK: Deleted $p") }
            else fb.add("REJECTED: $p")
            processDeletes(del, idx + 1, fb)
        }
    }

    private fun sendFeedback(fb: List<String>) {
        if (!running) return
        val msg = JSONObject().apply {
            put("role", "user")
            put("parts", JSONArray().put(JSONObject().put("text", prompt.buildFeedback(fb))))
        }
        history.put(msg)
        cb?.onStatusChanged("AI 계속 작업... ($iter/$MAX_ITER)")
        sendNext()
    }

    // ── 도구 실행 ──────────────────────────────────────────────

    private fun executeGrep(path: String, pattern: String, recursive: Boolean): String {
        val base = root ?: return "GREP FAIL: No project"
        return try {
            val target = File(base, path)
            if (!target.exists()) return "GREP FAIL: Path not found: $path"
            val regex = Regex(pattern)
            val sb    = StringBuilder("GREP RESULT ($pattern):\n")
            val files = if (target.isFile) sequenceOf(target) else target.walkTopDown().filter { it.isFile }
            var matchCount = 0
            for (f in files) {
                if (matchCount > 50) { sb.append("... (limit reached)"); break }
                try { f.forEachLine { line ->
                    if (regex.containsMatchIn(line)) { sb.append("${f.relativeTo(base).path}: $line\n"); matchCount++ }
                }} catch (_: Exception) {}
            }
            if (matchCount == 0) "GREP: No matches found." else sb.toString()
        } catch (e: Exception) { "GREP ERROR: ${e.message}" }
    }

    private fun executeReadRange(path: String, start: Int, end: Int): String {
        val base = root ?: return "READ FAIL: No project"
        return try {
            val f = File(base, path)
            if (!f.exists()) return "READ FAIL: File not found: $path"
            val lines = f.readLines()
            val s = (start - 1).coerceAtLeast(0)
            val e = end.coerceAtMost(lines.size)
            if (s >= e) return "READ: Empty range (file has ${lines.size} lines)"
            val sb = StringBuilder("READ $path ($start-$end):\n")
            for (i in s until e) sb.append("${i + 1}: ${lines[i]}\n")
            sb.toString()
        } catch (e: Exception) { "READ ERROR: ${e.message}" }
    }

    private fun executePatch(path: String, oldStr: String, newStr: String): String {
        val base = root ?: return "PATCH FAIL: No project"
        return try {
            val f = File(base, path)
            if (!f.exists()) return "PATCH FAIL: File not found: $path"
            val content = f.readText()
            if (!content.contains(oldStr)) return "PATCH FAIL: 'old' string not found. Ensure exact match including whitespace."
            f.writeText(content.replaceFirst(oldStr, newStr))
            cb?.onFileUpdated(path, true, "Patched")
            "PATCH OK: $path updated."
        } catch (e: Exception) { "PATCH ERROR: ${e.message}" }
    }

    // ── JSON 추출 (중첩 지원) ──────────────────────────────────

    private fun extractJson(text: String): String? {
        // 코드 블록에서 추출 시도 (그리디)
        val codeBlock = Regex("```(?:json)?\\s*\\n?(\\{[\\s\\S]*\\})\\s*```").find(text)
        if (codeBlock != null) {
            val c = codeBlock.groupValues[1]
            return try { JSONObject(c); c } catch (_: Exception) { null }
        }
        // 괄호 깊이 카운팅으로 추출 (중첩 JSON 대응)
        val start = text.indexOf('{')
        if (start == -1) return null
        var depth = 0; var inString = false; var escape = false
        for (i in start until text.length) {
            val c = text[i]
            if (escape) { escape = false; continue }
            if (c == '\\' && inString) { escape = true; continue }
            if (c == '"') { inString = !inString; continue }
            if (!inString) when (c) {
                '{' -> depth++
                '}' -> { depth--; if (depth == 0) {
                    val candidate = text.substring(start, i + 1)
                    return try { JSONObject(candidate); candidate } catch (_: Exception) { null }
                }}
            }
        }
        return null
    }

    // ── 메모리 트리밍용 단발 API 요약 요청 (히스토리에 저장 안 함) ──

    private fun summarizeBlocking(text: String): String? {
        val config = apiConfigs.getOrNull(currentApiIdx) ?: return null
        val summaryPrompt = "다음 대화 기록을 한 줄로 요약해주세요 (50자 이내, 한국어):\n$text"
        val body = JSONObject().apply {
            put("contents", JSONArray().put(JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray().put(JSONObject().put("text", summaryPrompt)))
            }))
            put("generationConfig", JSONObject().apply { put("maxOutputTokens", 120) })
        }.toString().toRequestBody("application/json".toMediaType())
        val url = "https://generativelanguage.googleapis.com/v1beta/models/${config.model}:generateContent?key=${config.key}"
        return try {
            val resp = client.newCall(Request.Builder().url(url).post(body).build()).execute()
            val raw  = resp.body?.string() ?: return null
            JSONObject(raw).optJSONArray("candidates")?.optJSONObject(0)
                ?.optJSONObject("content")?.optJSONArray("parts")
                ?.optJSONObject(0)?.optString("text")?.trim()
        } catch (_: Exception) { null }
    }

    // ── 세션 저장/로드 ─────────────────────────────────────────

    private fun saveSession() {
        val mgr = memoryMgr ?: return
        try { mgr.save(MemoryManager.Session(archiveSummary, history, localConvHistory.toList())) } catch (_: Exception) {}
    }

    private fun loadSession() {
        val mgr = memoryMgr ?: run { history = JSONArray(); return }
        val session    = mgr.load()
        history        = session.messages
        archiveSummary = session.archiveSummary
        localConvHistory.clear()
        localConvHistory.addAll(session.localHistory)
    }

    private fun recentLocalHistory(): List<Pair<String, String>> {
        var total = 0
        val result = mutableListOf<Pair<String, String>>()
        for ((u, a) in localConvHistory.reversed()) {
            val size = u.length + a.length + 50
            if (total + size > 600) break
            result.add(0, u to a)
            total += size
        }
        return result
    }

    private fun addLocalHistory(user: String, assistant: String) {
        val clean = assistant.replace(Regex("<think>[\\s\\S]*?</think>\\s*"), "").trim()
        if (clean.isEmpty()) return
        // 소스 태그 포함 (기억 파일에 어떤 AI가 답변했는지 기록)
        val tagged = "[Local AI: $localModelName] ${clean.take(190)}"
        localConvHistory.add(user.take(200) to tagged)
        if (localConvHistory.size > 3) localConvHistory.removeAt(0)
    }

    // ── 로컬 AI 도구 실행 루프 ────────────────────────────────
    private fun executeLocalAI(local: LocalAIManager, prompt: String, r: File, iteration: Int) {
        if (iteration >= MAX_ITER) {
            running = false
            cb?.onError("Local AI: 최대 반복(${MAX_ITER}) 도달")
            return
        }

        // 첫 번째 iteration: 프로젝트 컨텍스트를 서비스에 주입 (시스템 프롬프트에 포함됨)
        if (iteration == 0) {
            val ctx = PromptBuilder(800).buildLocalContext(r, prompt)
            local.setContext(ctx)
        }

        local.setStreamCallback(object : LocalAIManager.StreamCallback {
            override fun onToken(token: String) {}

            override fun onComplete(fullResponse: String) {
                val json = extractJson(fullResponse)
                if (json == null) {
                    // 일반 텍스트 응답 — 완료
                    if (iteration == 0) addLocalHistory(prompt, fullResponse)
                    saveSession()
                    running = false
                    cb?.onCompleted(fullResponse)
                    return
                }
                try {
                    val result = JSONObject(json)
                    val thought = result.optString("thought", "")
                    if (thought.isNotEmpty()) mainHandler { cb?.onThought(thought) }

                    val feedback = mutableListOf<String>()
                    val grep      = result.optJSONObject("grep")
                    val readRange = result.optJSONObject("read_range")
                    val patch     = result.optJSONObject("patch")
                    val updates   = result.optJSONArray("updates")
                    val deletes   = result.optJSONArray("deletes")

                    if (grep != null) feedback.add(executeGrep(grep.optString("path","."), grep.optString("pattern",""), grep.optBoolean("recursive",true)))
                    if (readRange != null) feedback.add(executeReadRange(readRange.optString("path"), readRange.optInt("start",1), readRange.optInt("end",100)))
                    if (patch != null) feedback.add(executePatch(patch.optString("path"), patch.optString("old"), patch.optString("new")))

                    if (updates != null && updates.length() > 0) {
                        for (i in 0 until updates.length()) {
                            val o = updates.getJSONObject(i)
                            val p = o.getString("path"); val c = o.getString("content")
                            try {
                                val f = File(r, p); f.parentFile?.mkdirs(); f.writeText(c)
                                feedback.add("OK: Wrote $p")
                                mainHandler { cb?.onFileUpdated(p, true, "완료") }
                            } catch (e: Exception) { feedback.add("FAIL: $p - ${e.message}") }
                        }
                    }

                    if (deletes != null && deletes.length() > 0) {
                        for (i in 0 until deletes.length()) {
                            val p = deletes.getString(i)
                            cb?.onDeleteRequested(p) { ok ->
                                if (ok) { File(r, p).deleteRecursively(); feedback.add("OK: Deleted $p") }
                                else feedback.add("REJECTED: $p")
                                continueLocalAI(local, prompt, r, iteration, feedback)
                            }
                        }
                        return // continueLocalAI called from delete callback
                    }

                    if (feedback.isNotEmpty()) {
                        continueLocalAI(local, prompt, r, iteration, feedback)
                    } else {
                        // 도구 없음 — 완료
                        if (iteration == 0) addLocalHistory(prompt, fullResponse)
                        saveSession()
                        running = false
                        cb?.onCompleted(thought.ifEmpty { fullResponse })
                    }
                } catch (e: Exception) {
                    if (iteration == 0) addLocalHistory(prompt, fullResponse)
                    saveSession()
                    running = false
                    cb?.onCompleted(fullResponse)
                }
            }

            override fun onError(error: String) {
                logger.logError("Local AI Error: $error")
                running = false
                cb?.onError("Local AI: $error")
            }
        })
        local.sendPrompt(prompt)
    }

    private fun continueLocalAI(local: LocalAIManager, originalCommand: String, r: File, iteration: Int, feedback: List<String>) {
        if (!running) return
        val feedbackPrompt = "## Tool Results\n${feedback.joinToString("\n")}\n\nContinue if needed. If done, return thought with summary."
        mainHandler { cb?.onStatusChanged("Local AI 계속 작업… (${iteration + 1}/$MAX_ITER)") }
        executeLocalAI(local, feedbackPrompt, r, iteration + 1)
    }

    // ── 모델 목록 조회 ─────────────────────────────────────────

    fun fetchModels(key: String, onResult: (List<String>) -> Unit) {
        val req = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models?key=$key").get().build()
        client.newCall(req).enqueue(object : okhttp3.Callback {
            override fun onResponse(call: Call, resp: Response) {
                try {
                    val models = JSONObject(resp.body?.string() ?: "{}").getJSONArray("models")
                    val list   = mutableListOf<String>()
                    for (i in 0 until models.length()) {
                        val n = models.getJSONObject(i).getString("name")
                        if (n.contains("gemini")) list.add(n.removePrefix("models/"))
                    }
                    list.sortByDescending { it }
                    onResult(list)
                } catch (_: Exception) {
                    onResult(listOf("gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-pro"))
                }
            }
            override fun onFailure(call: Call, e: IOException) {
                onResult(listOf("gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-pro"))
            }
        })
    }

    private fun mainHandler(action: () -> Unit) {
        android.os.Handler(android.os.Looper.getMainLooper()).post { action() }
    }
}
