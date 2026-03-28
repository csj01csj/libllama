package com.aicode.studio.util

import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.ScrollView
import android.widget.TextView

class LogManager(private val tv: TextView, private val sv: ScrollView) {
    private val handler = Handler(Looper.getMainLooper())

    enum class Level(val tag: String) {
        INFO("[INFO]"), WARN("[WARN]"), ERROR("[ERR!]"),
        AI("[ AI ]"), BUILD("[BLD ]"), SYSTEM("[SYS ]"), USER("[나  ]")
    }

    private var lastMsg: String? = null
    private var lastLevel: Level? = null
    private var repeatCount = 1
    private var lastLineStart = 0

    fun log(msg: String, level: Level = Level.INFO) {
        handler.post {
            if (msg == lastMsg && level == lastLevel) {
                repeatCount++
                val newLastLine = "${level.tag} $msg x$repeatCount"
                val currentText = tv.editableText
                if (currentText != null) {
                    currentText.replace(lastLineStart, currentText.length, newLastLine)
                }
            } else {
                lastMsg = msg
                lastLevel = level
                repeatCount = 1
                if (tv.text.isNotEmpty()) tv.append("\n")
                lastLineStart = tv.text.length
                tv.append("${level.tag} $msg")
                trimIfNeeded()
            }
            sv.post { sv.fullScroll(View.FOCUS_DOWN) }
        }
    }

    fun clear() = handler.post { tv.text = "> AIDE Engine Ready" }
    fun logBuild(m: String) = log(m, Level.BUILD)
    fun logAI(m: String) = log(m, Level.AI)
    fun logError(m: String) = log(m, Level.ERROR)
    fun logSystem(m: String) = log(m, Level.SYSTEM)
    fun logUser(m: String) = log(m, Level.USER)

    private fun trimIfNeeded() {
        val lines = tv.text.lines()
        if (lines.size > 500) tv.text = lines.takeLast(400).joinToString("\n")
    }
}