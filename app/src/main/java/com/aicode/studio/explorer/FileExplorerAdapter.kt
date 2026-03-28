package com.aicode.studio.explorer

import android.graphics.Color
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File

class FileExplorerAdapter(
    private val onFileClick: (File) -> Unit,
    private val onFolderToggle: (File) -> Unit,
    private val onLongClick: (File, Int) -> Boolean
) : BaseAdapter() {

    private data class Node(val file: File, val depth: Int)

    private var items = mutableListOf<Node>()
    private val expanded = mutableSetOf<String>()

    fun setRoot(root: File) {
        items.clear()
        buildTree(root, 0, isRoot = true)
        notifyDataSetChanged()
    }

    fun toggleFolder(file: File) {
        val p = file.absolutePath
        if (expanded.contains(p)) expanded.remove(p) else expanded.add(p)
    }

    fun isExpanded(file: File) = expanded.contains(file.absolutePath)

    private fun buildTree(cur: File, depth: Int, isRoot: Boolean = false) {
        if (!isRoot) items.add(Node(cur, depth))
        if (cur.isDirectory && (isRoot || expanded.contains(cur.absolutePath))) {
            cur.listFiles()
                ?.filter { !it.name.startsWith(".") }
                ?.sortedWith(compareBy({ it.isFile }, { it.name.lowercase() }))
                ?.forEach { buildTree(it, if (isRoot) 0 else depth + 1) }
        }
    }

    override fun getCount() = items.size
    override fun getItem(pos: Int): File = items[pos].file
    override fun getItemId(pos: Int) = pos.toLong()

    override fun getView(pos: Int, cv: View?, parent: ViewGroup): View {
        val n = items[pos]
        val c = parent.context
        val row = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(12 + n.depth * 28, 10, 12, 10)
            gravity = Gravity.CENTER_VERTICAL
        }

        if (n.file.isDirectory) {
            row.addView(TextView(c).apply {
                text = if (expanded.contains(n.file.absolutePath)) "▼ " else "▶ "
                setTextColor(safeColor("#666666")); textSize = 11f
            })
        }

        val icon = when {
            n.file.isDirectory -> "📁"
            else -> when (n.file.extension.lowercase()) {
                "java" -> "☕"; "kt" -> "🟣"; "xml" -> "📐"
                "gradle" -> "🐘"; "json" -> "📋"; else -> "📄"
            }
        }
        val clr = when {
            n.file.isDirectory -> "#EBCB8B"
            else -> when (n.file.extension.lowercase()) {
                "java" -> "#E06C75"; "kt" -> "#C678DD"; "xml" -> "#61AFEF"
                "gradle" -> "#98C379"; "json" -> "#E5C07B"; else -> "#ABB2BF"
            }
        }

        row.addView(TextView(c).apply {
            text = "$icon ${n.file.name}"
            setTextColor(safeColor(clr)); textSize = 13f
            if (n.file.isDirectory) setTypeface(null, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })

        if (n.file.isFile) {
            row.addView(TextView(c).apply {
                text = fmtSize(n.file.length())
                setTextColor(safeColor("#555")); textSize = 9f
            })
        }

        row.setOnClickListener {
            if (n.file.isDirectory) { toggleFolder(n.file); onFolderToggle(n.file) }
            else onFileClick(n.file)
        }
        row.setOnLongClickListener { onLongClick(n.file, pos) }
        return row
    }

    private fun safeColor(c: String): Int {
        return try {
            Color.parseColor(c)
        } catch (e: Exception) {
            Color.WHITE
        }
    }

    private fun fmtSize(b: Long) = when {
        b < 1024 -> "${b}B"; b < 1048576 -> "${b/1024}K"
        else -> "${"%.1f".format(b/1048576.0)}M"
    }
}