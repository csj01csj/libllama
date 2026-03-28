package com.aicode.studio.editor

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.AttributeSet
import android.view.Gravity
import androidx.appcompat.widget.AppCompatEditText
import java.io.File

class CodeEditorView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
    defStyle: Int = android.R.attr.editTextStyle
) : AppCompatEditText(context, attrs, defStyle) {

    private val gutterPaint = Paint().apply {
        color = Color.parseColor("#5C6370"); textSize = 30f
        typeface = Typeface.MONOSPACE; textAlign = Paint.Align.RIGHT; isAntiAlias = true
    }
    private val gutterBg = Paint().apply { color = Color.parseColor("#1A1A2E") }
    private val curLineBg = Paint().apply { color = Color.parseColor("#2C313A") }
    private val divPaint = Paint().apply { color = Color.parseColor("#3E4451"); strokeWidth = 2f }

    private var gutterW = 90
    private var fileType = SyntaxHighlighter.FileType.UNKNOWN
    private var curFile: File? = null
    private var highlighting = false

    val isBusy get() = highlighting

    private val hlHandler = Handler(Looper.getMainLooper())
    private var hlRunnable: Runnable? = null

    var onContentChanged: ((String) -> Unit)? = null
    var onCursorMoved: ((Int, Int) -> Unit)? = null  // line, col

    init {
        setBackgroundColor(Color.parseColor("#1E1E2E"))
        setTextColor(Color.parseColor("#D4D4D4"))
        typeface = Typeface.MONOSPACE; textSize = 13f
        gravity = Gravity.TOP or Gravity.START
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        setHorizontallyScrolling(true)
        isVerticalScrollBarEnabled = true; isHorizontalScrollBarEnabled = true
        updateGutter()

        addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, before: Int, count: Int) {
                if (highlighting) return
                if (count == 1 && s != null && st < s.length && s[st] == '\n')
                    post { autoIndent(st) }
            }
            override fun afterTextChanged(s: Editable?) {
                if (highlighting) return
                updateGutter(); scheduleHighlight()
                onContentChanged?.invoke(s?.toString() ?: "")
            }
        })
    }

    override fun onSelectionChanged(selStart: Int, selEnd: Int) {
        super.onSelectionChanged(selStart, selEnd)
        if (layout != null && selStart >= 0) {
            val line = layout.getLineForOffset(selStart)
            val col = selStart - layout.getLineStart(line)
            onCursorMoved?.invoke(line + 1, col + 1)
        }
        invalidate()
    }

    fun openFile(file: File) {
        curFile = file
        fileType = SyntaxHighlighter.detect(file.name)
        highlighting = true
        setText(file.readText())
        highlighting = false
        applyHighlight()
        setSelection(0)
    }

    fun clearEditor() {
        highlighting = true
        curFile = null
        fileType = SyntaxHighlighter.FileType.UNKNOWN
        setText("")
        highlighting = false
    }

    fun saveFile(): Boolean {
        val f = curFile ?: return false
        return try { f.writeText(text?.toString() ?: ""); true } catch (_: Exception) { false }
    }

    fun setExternal(content: String) {
        highlighting = true; setText(content); highlighting = false; applyHighlight()
    }

    fun getFile(): File? = curFile

    fun hasFile() = curFile != null

    fun getFileType() = fileType

    // ─── Draw ────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        val l = layout ?: run { super.onDraw(canvas); return }
        val lc = lineCount.coerceAtLeast(1)

        // 거터 배경
        canvas.drawRect(scrollX.toFloat(), scrollY.toFloat(),
            (scrollX + gutterW - 8).toFloat(), (scrollY + height).toFloat(), gutterBg)

        // 현재 줄
        val sel = selectionStart
        if (sel >= 0) {
            val cl = l.getLineForOffset(sel)
            if (cl in 0 until lc)
                canvas.drawRect(scrollX.toFloat(), l.getLineTop(cl).toFloat(),
                    (scrollX + width).toFloat(), l.getLineBottom(cl).toFloat(), curLineBg)
        }

        // 줄 번호 (보이는 범위만)
        val fv = l.getLineForVertical(scrollY)
        val lv = l.getLineForVertical(scrollY + height)
        for (i in fv..minOf(lv + 1, lc - 1)) {
            canvas.drawText("${i + 1}", (scrollX + gutterW - 18).toFloat(),
                l.getLineBaseline(i).toFloat(), gutterPaint)
        }

        // 구분선
        val dx = (scrollX + gutterW - 6).toFloat()
        canvas.drawLine(dx, scrollY.toFloat(), dx, (scrollY + height).toFloat(), divPaint)

        super.onDraw(canvas)
    }

    private fun updateGutter() {
        val d = lineCount.toString().length.coerceAtLeast(2)
        val nw = d * 20 + 36
        if (nw != gutterW) { gutterW = nw; setPadding(gutterW, paddingTop, paddingRight, paddingBottom) }
    }

    // ─── Highlighting ────────────────────────────

    private fun scheduleHighlight() {
        hlRunnable?.let { hlHandler.removeCallbacks(it) }
        hlRunnable = Runnable { applyHighlight() }
        hlHandler.postDelayed(hlRunnable!!, 250)
    }

    private fun applyHighlight() {
        if (fileType == SyntaxHighlighter.FileType.UNKNOWN) return
        val e = text ?: return
        highlighting = true
        try { SyntaxHighlighter.highlight(e, fileType) } catch (_: Exception) {}
        highlighting = false
    }

    // ─── Auto-indent ─────────────────────────────

    private fun autoIndent(nlPos: Int) {
        val c = text?.toString() ?: return
        val ls = c.lastIndexOf('\n', nlPos - 1).let { if (it < 0) 0 else it + 1 }
        val line = c.substring(ls, nlPos)
        val indent = line.takeWhile { it == ' ' || it == '\t' }
        val extra = if (line.trimEnd().let { it.endsWith("{") || it.endsWith(">") || it.endsWith("(") }) "    " else ""
        val ins = indent + extra
        if (ins.isNotEmpty()) { highlighting = true; text?.insert(nlPos + 1, ins); highlighting = false }
    }
}