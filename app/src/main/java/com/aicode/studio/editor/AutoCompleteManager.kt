package com.aicode.studio.editor

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*

/**
 * 코드 자동완성 매니저
 * - Java/Kotlin 키워드, Android API 자동완성
 * - 현재 파일 심볼 수집
 * - 팝업 윈도우로 표시
 */
class AutoCompleteManager(private val ctx: Context, private val editor: CodeEditorView) {

    private var popup: PopupWindow? = null
    private val suggestions = mutableListOf<String>()

    // Android API 자주 쓰는 것들
    private val baseCompletions = listOf(
        // Java/Kotlin keywords
        "abstract", "boolean", "break", "byte", "case", "catch", "char", "class",
        "continue", "default", "do", "double", "else", "enum", "extends", "final",
        "finally", "float", "for", "if", "implements", "import", "instanceof", "int",
        "interface", "long", "new", "null", "package", "private", "protected", "public",
        "return", "short", "static", "super", "switch", "synchronized", "this", "throw",
        "throws", "try", "void", "while", "true", "false",
        "fun", "val", "var", "when", "object", "companion", "data", "sealed",
        "override", "open", "lateinit", "lazy", "suspend", "inline",
        // Android common
        "Activity", "AppCompatActivity", "Fragment", "Intent", "Bundle", "Context",
        "View", "ViewGroup", "TextView", "EditText", "Button", "ImageView", "ImageButton",
        "LinearLayout", "RelativeLayout", "FrameLayout", "ConstraintLayout", "ScrollView",
        "RecyclerView", "Adapter", "ViewHolder", "LayoutInflater",
        "Toast", "Log", "Handler", "Looper", "Thread", "Runnable",
        "SharedPreferences", "AlertDialog", "Menu", "MenuItem",
        "OnClickListener", "OnLongClickListener", "TextWatcher",
        "onCreate", "onStart", "onResume", "onPause", "onStop", "onDestroy",
        "onCreateView", "onViewCreated",
        "findViewById", "setContentView", "getIntent", "startActivity",
        "getString", "getResources", "getSystemService",
        "setOnClickListener", "setAdapter", "notifyDataSetChanged",
        "setText", "getText", "setVisibility", "setEnabled",
        "VISIBLE", "INVISIBLE", "GONE",
        "match_parent", "wrap_content",
        "R.layout", "R.id", "R.string", "R.drawable", "R.color",
        "android.widget", "android.view", "android.os", "android.app",
        "android.content", "android.graphics", "android.util",
        "androidx.appcompat", "androidx.core", "androidx.recyclerview",
        "String", "Int", "Long", "Float", "Double", "Boolean", "List", "Map", "Set",
        "ArrayList", "HashMap", "HashSet", "MutableList", "MutableMap",
        "IOException", "Exception", "RuntimeException", "IllegalArgumentException",
        "System.out.println", "Log.d", "Log.e", "Log.i", "Log.w",
        "runOnUiThread", "post", "postDelayed"
    )

    // 현재 파일에서 식별자 수집
    private fun collectLocalSymbols(): List<String> {
        val text = editor.text?.toString() ?: return emptyList()
        val pattern = Regex("\\b[A-Za-z_][A-Za-z0-9_]{2,}\\b")
        return pattern.findAll(text).map { it.value }.distinct().toList()
    }

    fun showSuggestions(prefix: String) {
        if (prefix.length < 2) { dismiss(); return }

        val local = collectLocalSymbols()
        val all = (local + baseCompletions).distinct()
        suggestions.clear()
        suggestions.addAll(
            all.filter { it.startsWith(prefix, ignoreCase = true) && it != prefix }
                .sortedBy { it.length }
                .take(8)
        )

        if (suggestions.isEmpty()) { dismiss(); return }

        val listView = ListView(ctx).apply {
            adapter = object : ArrayAdapter<String>(ctx, 0, suggestions) {
                override fun getView(pos: Int, cv: View?, parent: ViewGroup): View {
                    return (cv as? TextView ?: TextView(ctx).apply {
                        setPadding(16, 8, 16, 8)
                        textSize = 13f
                        setTextColor(Color.parseColor("#D4D4D4"))
                    }).apply { text = getItem(pos) }
                }
            }
            setBackgroundColor(Color.parseColor("#2D2D30"))
            divider = ColorDrawable(Color.parseColor("#3E3E42"))
            dividerHeight = 1
            setOnItemClickListener { _, _, pos, _ ->
                insertCompletion(suggestions[pos], prefix)
                dismiss()
            }
        }

        popup?.dismiss()
        popup = PopupWindow(listView, 300, ViewGroup.LayoutParams.WRAP_CONTENT, true).apply {
            setBackgroundDrawable(ColorDrawable(Color.parseColor("#2D2D30")))
            elevation = 8f
            isFocusable = false  // 에디터 포커스 유지
            isOutsideTouchable = true
        }

        // 커서 위치 기준으로 팝업 표시
        try {
            val pos = editor.selectionStart
            val layout = editor.layout ?: return
            val line = layout.getLineForOffset(pos)
            val x = layout.getPrimaryHorizontal(pos).toInt()
            val y = layout.getLineBottom(line)
            popup?.showAtLocation(editor, Gravity.NO_GRAVITY,
                x + editor.paddingLeft, y - editor.scrollY + editor.top)
        } catch (_: Exception) {
            popup?.showAsDropDown(editor, 100, -200)
        }
    }

    private fun insertCompletion(completion: String, prefix: String) {
        val start = editor.selectionStart - prefix.length
        val end = editor.selectionStart
        if (start >= 0) {
            editor.text?.replace(start, end, completion)
        }
    }

    fun dismiss() {
        popup?.dismiss(); popup = null
    }

    fun isShowing() = popup?.isShowing == true
}