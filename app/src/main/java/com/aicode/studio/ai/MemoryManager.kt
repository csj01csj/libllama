package com.aicode.studio.ai

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * 프로젝트별 AI 대화 기억 관리.
 *
 * 파일 구조:
 *   .ai_session.json         → 현재 활성 대화 (archive_summary + messages)
 *   .ai_memory/archive_001.json  → 잘려나간 과거 대화 원본
 *   .ai_memory/archive_002.json  → ...
 *
 * archive_summary 형식 (session 맨 앞):
 *   [{seq, time, summary, file}, ...]
 *   → AI가 과거 기억 꺼낼 때 file 경로로 아카이브 읽기 가능
 */
class MemoryManager(private val root: File) {

    companion object {
        const val SESSION_FILE  = ".ai_session.json"
        const val ARCHIVE_DIR   = ".ai_memory"
        const val MAX_CHARS     = 80_000   // 이 이상이면 트리밍
        const val KEEP_HEAD     = 1        // 맨 앞 N개 메시지 (초기 시스템 설정) 유지
        const val KEEP_RECENT   = 10       // 최근 N개 메시지 항상 유지
    }

    private val sessionFile get() = File(root, SESSION_FILE)
    private val archiveDir  get() = File(root, ARCHIVE_DIR)

    data class ArchiveEntry(
        val seq      : Int,
        val time     : String,
        val summary  : String,
        val fileName : String   // 프로젝트 루트 기준 상대 경로
    )

    data class Session(
        val archiveSummary: MutableList<ArchiveEntry> = mutableListOf(),
        val messages      : JSONArray = JSONArray(),
        val localHistory  : List<Pair<String, String>> = emptyList()
    )

    // ── 로드 ──────────────────────────────────────────────────
    fun load(): Session {
        if (!sessionFile.exists()) return Session()
        return try {
            val text = sessionFile.readText()
            if (text.trimStart().startsWith("[")) {
                // 구 형식 (JSONArray) → 자동 마이그레이션
                Session(messages = JSONArray(text))
            } else {
                val obj = JSONObject(text)
                val summaries = mutableListOf<ArchiveEntry>()
                val arr = obj.optJSONArray("archive_summary") ?: JSONArray()
                for (i in 0 until arr.length()) {
                    val e = arr.getJSONObject(i)
                    summaries.add(ArchiveEntry(
                        seq      = e.getInt("seq"),
                        time     = e.getString("time"),
                        summary  = e.getString("summary"),
                        fileName = e.getString("file")
                    ))
                }
                val localHistory = mutableListOf<Pair<String, String>>()
                val lhArr = obj.optJSONArray("local_history") ?: JSONArray()
                for (i in 0 until lhArr.length()) {
                    val e = lhArr.getJSONObject(i)
                    localHistory.add(e.getString("u") to e.getString("a"))
                }
                Session(summaries, obj.optJSONArray("messages") ?: JSONArray(), localHistory)
            }
        } catch (_: Exception) { Session() }
    }

    // ── 저장 ──────────────────────────────────────────────────
    fun save(session: Session) {
        try {
            val arr = JSONArray()
            session.archiveSummary.forEach { e ->
                arr.put(JSONObject().apply {
                    put("seq",     e.seq)
                    put("time",    e.time)
                    put("summary", e.summary)
                    put("file",    e.fileName)
                })
            }
            val localArr = JSONArray()
            session.localHistory.forEach { (u, a) ->
                localArr.put(JSONObject().apply { put("u", u); put("a", a) })
            }
            val obj = JSONObject().apply {
                put("archive_summary", arr)
                put("messages", session.messages)
                put("local_history", localArr)
            }
            sessionFile.writeText(obj.toString())
        } catch (_: Exception) {}
    }

    // ── 트리밍 필요 여부 ──────────────────────────────────────
    fun needsTrimming(session: Session): Boolean =
        session.messages.toString().length > MAX_CHARS

    /**
     * 오래된 메시지를 잘라내고 아카이브 저장.
     *
     * @param summarizer 잘려나간 대화 텍스트를 받아 한 줄 요약 반환.
     *                   null이면 요약 없이 아카이브만 저장.
     * @return 업데이트된 Session
     */
    fun trimAndArchive(session: Session, summarizer: ((String) -> String?)? = null): Session {
        val msgs  = session.messages
        val total = msgs.length()
        if (total <= KEEP_HEAD + KEEP_RECENT) return session

        val cutFrom = KEEP_HEAD
        val cutTo   = total - KEEP_RECENT   // exclusive
        if (cutTo <= cutFrom) return session

        // 잘려나갈 메시지 추출
        val cutMsgs = JSONArray()
        for (i in cutFrom until cutTo) cutMsgs.put(msgs.getJSONObject(i))

        // 아카이브 파일 저장
        archiveDir.mkdirs()
        val seq         = session.archiveSummary.size + 1
        val archiveName = "$ARCHIVE_DIR/archive_%03d.json".format(seq)
        try { File(root, archiveName).writeText(cutMsgs.toString()) } catch (_: Exception) {}

        // AI에게 한 줄 요약 요청 (이 요청은 history에 저장 안 함)
        val summaryText = summarizer?.invoke(buildTextForSummary(cutMsgs))
            ?: "${cutMsgs.length()}개 대화 아카이브"
        val time = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date())

        // 새 메시지 배열 구성 (HEAD + 최근 N개만 유지)
        val newMsgs = JSONArray()
        for (i in 0 until KEEP_HEAD)   if (i < total) newMsgs.put(msgs.getJSONObject(i))
        for (i in cutTo until total)   newMsgs.put(msgs.getJSONObject(i))

        val newSummaries = session.archiveSummary.toMutableList()
        newSummaries.add(ArchiveEntry(seq, time, summaryText, archiveName))

        return Session(newSummaries, newMsgs)
    }

    /**
     * 시스템 프롬프트에 포함할 아카이브 인덱스 텍스트 반환.
     * 형식: "순서 - 시간 - 요약내용 - 파일이름"
     */
    fun buildArchiveIndexText(session: Session): String {
        if (session.archiveSummary.isEmpty()) return ""
        return session.archiveSummary.joinToString("\n") {
            "${it.seq} - ${it.time} - ${it.summary} - ${it.fileName}"
        }
    }

    /** 아카이브 파일 내용 읽기 (AI가 과거 기억 참조 요청 시) */
    fun readArchive(fileName: String): String? =
        try { File(root, fileName).readText() } catch (_: Exception) { null }

    // ── 내부: 요약 요청용 텍스트 ──────────────────────────────
    private fun buildTextForSummary(msgs: JSONArray): String {
        val sb = StringBuilder()
        for (i in 0 until msgs.length()) {
            val m    = msgs.getJSONObject(i)
            val role = m.optString("role", "?")
            val text = m.optJSONArray("parts")
                ?.optJSONObject(0)?.optString("text", "") ?: ""
            sb.appendLine("[$role]: ${text.take(300)}")
        }
        return sb.toString().trim()
    }
}
