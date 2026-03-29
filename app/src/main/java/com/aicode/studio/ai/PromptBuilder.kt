package com.aicode.studio.ai

import java.io.File

/**
 * AI 프롬프트 빌더
 * - 파일 트리 기반 컨텍스트
 * - 관련 파일만 선택적 전송 (토큰 예산 관리)
 * - 시스템/유저/피드백 프롬프트 분리
 */
class PromptBuilder(private val maxChars: Int = 60_000) {

    data class FileCtx(val path: String, val content: String, val size: Int)

    fun buildSystemPrompt(root: File, apiIndex: Int = 1, archiveSummary: String = ""): String {
        val tree = buildTree(root)
        val archiveSection = if (archiveSummary.isNotEmpty()) """

## Past Conversation Archive Index
The following is a summary index of archived past conversations. If you need to recall specific past context, reference the file name to retrieve it.
(Format: seq - time - summary - file_path)
$archiveSummary
""" else ""
        return """You are an expert Android developer agent inside a mobile IDE called Aicode Studio.
You are currently operating as API Key #$apiIndex.$archiveSection

## Communication
- **Respond in the same language the user is using.** If the user writes in Korean, reply in Korean. If in English, reply in English.
- Your "thought" field should be detailed and explain your plan to the user in the same language.
- **Log Format**: When working, your thoughts will be shown in parentheses, e.g., "(분석 중...)" or "(Analyzing...)". When finished, you must provide a final concise summary of your actions.

## Output Format
Output ONLY valid JSON. No markdown, no explanation outside JSON.
```json
{
  "thought": "(작업 계획 및 생각...)",
  "grep": {"path": "optional/dir", "pattern": "regex", "recursive": true},
  "read_range": {"path": "file/path", "start": 1, "end": 20},
  "patch": {"path": "file/path", "old": "exact string to replace", "new": "new string"},
  "updates": [{"path": "relative/path", "content": "FULL file content"}],
  "deletes": ["path/to/delete"]
}
```

## Tools
1. **grep**: Search for a regex pattern in files. Use this to find code locations.
2. **read_range**: Read specific lines of a file. Use this to read large files partially.
3. **patch**: Replace a specific string block with a new one. PREFER this over 'updates' for editing existing files to save tokens.
   - `old` must match the file content EXACTLY (including whitespace).
   - If `old` is not unique, the patch will fail.
4. **updates**: Overwrite the ENTIRE file. Use this for creating new files or when the file is small.

## Rules
1. "path" is relative to project root.
2. When done: {"thought": "완료: <요약>", "updates": [], "deletes": []}
3. Register new Activities in AndroidManifest.xml.
4. Include xml declaration in layout XML.
5. Create parent directories as needed.
6. **BATCH all operations**: Put ALL files to create/modify in one `updates` array and ALL files to delete in one `deletes` array — never spread across multiple responses.
7. **Do NOT generate binary or large data files** (images, animation JSON, encoded data). Reference external libraries instead.
8. You can only use ONE of grep/read_range/patch per response. Use `updates` for multiple file writes.

## Project Structure
```
$tree
```"""
    }

    fun buildUserPrompt(command: String, root: File, explicitFiles: List<File>? = null): String {
        val allFiles = collectFiles(root)
        val selected = selectFiles(command, allFiles, explicitFiles, root)
        val ctx = if (selected.isNotEmpty()) {
            "\n\n## Relevant Source Files\n" + selected.joinToString("\n\n") {
                "=== ${it.path} ===\n${it.content}"
            }
        } else ""
        return "## Request\n$command$ctx"
    }

    fun buildFeedback(results: List<String>): String =
        "## System Report\n${results.joinToString("\n")}\n\nContinue if needed. If done, return thought with summary and empty updates."

    // ─── Tree Builder ─────────────────────────────

    /** 로컬 AI용 경량 컨텍스트: 파일 트리 + 가장 관련성 높은 파일 1개 내용 (토큰 예산 절약) */
    fun buildLocalContext(root: File, command: String, maxCharsCtx: Int = 800): String {
        val tree = buildTree(root)
        val sb = StringBuilder("## Project Files\n```\n${tree.take(600)}\n```")
        val all = collectFiles(root)
        val cmdLow = command.lowercase()
        val top = all.map { it to score(it, cmdLow) }.filter { it.second > 0 }
            .maxByOrNull { it.second }?.first
        if (top != null && sb.length < maxCharsCtx) {
            val rel = top.relativeTo(root).path
            val content = top.readText().take(maxCharsCtx - sb.length - 60)
            sb.append("\n\n## Key File: $rel\n```\n$content\n```")
        }
        return sb.toString()
    }

    internal fun buildTree(root: File, prefix: String = "", depth: Int = 0): String {
        if (depth > 5) return "${prefix}...\n"
        val sb = StringBuilder()
        val kids = root.listFiles()
            ?.filter { !it.name.startsWith(".") && it.name != "build" }
            ?.sortedWith(compareBy({ it.isFile }, { it.name })) ?: return ""
        for ((i, f) in kids.withIndex()) {
            val last = i == kids.size - 1
            val conn = if (last) "└── " else "├── "
            sb.appendLine("$prefix$conn${f.name}${if (f.isFile) " (${fmtSize(f.length())})" else ""}")
            if (f.isDirectory) sb.append(buildTree(f, prefix + if (last) "    " else "│   ", depth + 1))
        }
        return sb.toString()
    }

    // ─── File Selection (Token Budget) ────────────

    private fun collectFiles(root: File): List<File> =
        root.walkTopDown()
            .filter { it.isFile && !it.path.contains("/build/") && !it.name.startsWith(".") }
            .filter { it.extension.lowercase() in listOf("java","kt","xml","gradle","json","properties") }
            .toList()

    private fun selectFiles(cmd: String, all: List<File>, explicit: List<File>?, root: File): List<FileCtx> {
        val cmdLow = cmd.lowercase()
        val result = mutableListOf<FileCtx>()
        var budget = maxChars

        // 항상 포함: Manifest
        all.find { it.name == "AndroidManifest.xml" }?.let {
            val fc = toCtx(it, root); result.add(fc); budget -= fc.size
        }

        // 명시적 파일
        explicit?.forEach { f ->
            if (budget > 0 && f.exists()) {
                val fc = toCtx(f, root)
                if (fc.size <= budget) { result.add(fc); budget -= fc.size }
            }
        }

        // 관련성 스코어링
        val scored = all
            .filter { f -> result.none { it.path == f.relativeTo(root).path } }
            .map { it to score(it, cmdLow) }
            .filter { it.second > 0 }
            .sortedByDescending { it.second }

        for ((f, _) in scored) {
            if (budget <= 0) break
            val fc = toCtx(f, root)
            if (fc.size <= budget) { result.add(fc); budget -= fc.size }
        }

        // 남은 예산으로 나머지 파일
        if (budget > 2000) {
            all.filter { f -> result.none { it.path == f.relativeTo(root).path } }
                .sortedBy { it.length() }
                .forEach { f ->
                    if (budget <= 0) return@forEach
                    val fc = toCtx(f, root)
                    if (fc.size <= budget) { result.add(fc); budget -= fc.size }
                }
        }
        return result
    }

    private fun score(f: File, cmd: String): Int {
        var s = 0
        val n = f.nameWithoutExtension.lowercase()
        if (cmd.contains(n)) s += 10
        val kwMap = mapOf(
            "login" to listOf("login","auth","sign"), "main" to listOf("main","메인","홈"),
            "layout" to listOf("xml","layout","ui","화면","뷰"), "manifest" to listOf("permission","권한","activity"),
            "list" to listOf("list","리스트","recycler","adapter","목록"),
            "network" to listOf("api","http","retrofit","네트워크","서버"),
            "data" to listOf("db","data","room","sqlite","저장","데이터")
        )
        for ((_, kws) in kwMap) {
            if (kws.any { cmd.contains(it) } && kws.any { n.contains(it) }) s += 5
        }
        if (f.name == "build.gradle") s += 3
        if (n.contains("main")) s += 2
        if (f.extension in listOf("java","kt")) s += 1
        return s
    }

    private fun toCtx(f: File, root: File): FileCtx {
        val c = try { f.readText() } catch (_: Exception) { "// read error" }
        return FileCtx(f.relativeTo(root).path, c, c.length)
    }

    private fun fmtSize(b: Long) = when {
        b < 1024 -> "${b}B"; b < 1048576 -> "${b/1024}KB"; else -> "${"%.1f".format(b/1048576.0)}MB"
    }
}