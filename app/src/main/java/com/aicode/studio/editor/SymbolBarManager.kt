package com.aicode.studio.editor

import android.graphics.Color
import android.view.View
import android.widget.Button
import android.widget.LinearLayout

/**
 * 키보드 위 심볼 바 - 자주 쓰는 코드 심볼 빠른 입력
 * Tab, {}, (), [], <>, ;, :, =, ", /, @, . 등
 */
class SymbolBarManager(
    private val container: LinearLayout,
    private val editor: CodeEditorView
) {
    private val javaSymbols = listOf(
        "→" to "\t", "{" to "{", "}" to "}", "(" to "(", ")" to ")",
        "[" to "[", "]" to "]", "<" to "<", ">" to ">",
        ";" to ";", ":" to ":", "=" to "=", "\"" to "\"",
        "." to ".", "," to ",", "!" to "!", "@" to "@",
        "/" to "/", "+" to "+", "-" to "-", "_" to "_",
        "&&" to "&&", "||" to "||", "->" to "->", "::" to "::"
    )

    private val xmlSymbols = listOf(
        "→" to "\t", "<" to "<", ">" to ">", "/" to "/",
        "\"" to "\"", "=" to "=", ":" to ":", "." to ".",
        "</" to "</", "/>" to "/>", "<!--" to "<!-- ", "-->" to " -->",
        "android:" to "android:", "app:" to "app:",
        "match" to "match_parent", "wrap" to "wrap_content"
    )

    fun setup() {
        refresh(SyntaxHighlighter.FileType.JAVA)
    }

    fun refresh(fileType: SyntaxHighlighter.FileType) {
        container.removeAllViews()
        val symbols = when (fileType) {
            SyntaxHighlighter.FileType.XML -> xmlSymbols
            else -> javaSymbols
        }

        for ((label, insert) in symbols) {
            container.addView(Button(container.context).apply {
                text = label; textSize = 12f
                setTextColor(Color.parseColor("#D4D4D4"))
                setBackgroundColor(Color.parseColor("#3C3C3C"))
                setPadding(16, 4, 16, 4)
                minWidth = 0; minimumWidth = 0; minHeight = 0; minimumHeight = 0
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.MATCH_PARENT
                )
                lp.setMargins(2, 2, 2, 2)
                layoutParams = lp
                setOnClickListener { insertSymbol(insert) }
            })
        }
    }

    private fun insertSymbol(symbol: String) {
        val start = editor.selectionStart
        val end = editor.selectionEnd
        editor.text?.replace(start, end, symbol)
        // 괄호 자동 닫기
        val autoClose = mapOf("{" to "}", "(" to ")", "[" to "]", "<" to ">")
        autoClose[symbol]?.let { close ->
            val pos = editor.selectionStart
            editor.text?.insert(pos, close)
            editor.setSelection(pos)
        }
    }
}