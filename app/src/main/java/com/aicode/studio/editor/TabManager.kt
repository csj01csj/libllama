package com.aicode.studio.editor

import android.graphics.Color
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File

/**
 * 에디터 탭 매니저
 * - 여러 파일 동시 열기
 * - 탭 전환, 닫기
 * - 수정된 파일 표시 (●)
 */
class TabManager(
    private val container: LinearLayout,
    private val scrollView: HorizontalScrollView,
    private val onTabSelect: (File) -> Unit,
    private val onTabClose: (File) -> Unit
) {
    data class Tab(val file: File, var modified: Boolean = false)

    private val tabs = mutableListOf<Tab>()
    private var activeFile: File? = null

    fun openTab(file: File) {
        if (tabs.none { it.file.absolutePath == file.absolutePath }) {
            tabs.add(Tab(file))
        }
        activeFile = file
        refresh()
    }

    fun closeTab(file: File) {
        tabs.removeAll { it.file.absolutePath == file.absolutePath }
        if (activeFile?.absolutePath == file.absolutePath) {
            activeFile = tabs.lastOrNull()?.file
        }
        refresh()
        onTabClose(file)
        activeFile?.let { onTabSelect(it) }
    }

    fun markModified(file: File, modified: Boolean) {
        tabs.find { it.file.absolutePath == file.absolutePath }?.modified = modified
        refresh()
    }

    fun getActiveFile(): File? = activeFile

    fun getOpenFiles(): List<File> = tabs.map { it.file }

    fun closeAllTabs() {
        tabs.clear()
        activeFile = null
        refresh()
    }

    fun hasUnsaved(): Boolean = tabs.any { it.modified }

    private fun refresh() {
        container.removeAllViews()
        for (tab in tabs) {
            val isActive = tab.file.absolutePath == activeFile?.absolutePath
            val tv = TextView(container.context).apply {
                val prefix = if (tab.modified) "● " else ""
                text = "$prefix${tab.file.name}"
                textSize = 12f
                setTextColor(if (isActive) Color.WHITE else Color.parseColor("#888888"))
                setBackgroundColor(if (isActive) Color.parseColor("#1E1E1E") else Color.parseColor("#2D2D2D"))
                setPadding(24, 8, 24, 8)
                gravity = Gravity.CENTER
                if (isActive) setTypeface(null, Typeface.BOLD)

                setOnClickListener {
                    activeFile = tab.file
                    onTabSelect(tab.file)
                    refresh()
                }
                setOnLongClickListener {
                    closeTab(tab.file)
                    true
                }
            }

            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            lp.setMargins(0, 0, 1, 0) // 탭 구분선
            container.addView(tv, lp)
        }

        // 활성 탭이 보이도록 스크롤
        scrollView.post {
            val activeIdx = tabs.indexOfFirst { it.file.absolutePath == activeFile?.absolutePath }
            if (activeIdx >= 0 && activeIdx < container.childCount) {
                val child = container.getChildAt(activeIdx)
                scrollView.smoothScrollTo(child.left - 50, 0)
            }
        }
    }
}