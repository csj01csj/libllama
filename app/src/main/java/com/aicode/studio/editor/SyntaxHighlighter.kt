package com.aicode.studio.editor

import android.graphics.Color
import android.text.Spannable
import android.text.style.ForegroundColorSpan
import java.util.regex.Pattern

object SyntaxHighlighter {

    private val C_KW   = Color.parseColor("#C678DD")
    private val C_TYPE = Color.parseColor("#E5C07B")
    private val C_STR  = Color.parseColor("#98C379")
    private val C_CMT  = Color.parseColor("#5C6370")
    private val C_NUM  = Color.parseColor("#D19A66")
    private val C_ANN  = Color.parseColor("#61AFEF")
    private val C_TAG  = Color.parseColor("#E06C75")
    private val C_ATTR = Color.parseColor("#D19A66")
    private val C_XVAL = Color.parseColor("#98C379")

    private val KEYWORDS = listOf(
        "abstract","assert","boolean","break","byte","case","catch","char","class",
        "const","continue","default","do","double","else","enum","extends","final",
        "finally","float","for","goto","if","implements","import","instanceof","int",
        "interface","long","native","new","null","package","private","protected",
        "public","return","short","static","strictfp","super","switch","synchronized",
        "this","throw","throws","transient","try","void","volatile","while","true","false",
        "fun","val","var","when","is","in","object","companion","data","sealed","inline",
        "reified","suspend","override","open","internal","lateinit","lazy","by","init",
        "constructor","typealias","it","as","operator","infix","tailrec"
    )
    private val TYPES = listOf(
        "String","Int","Long","Float","Double","Boolean","Byte","Char","List","Map",
        "Set","Array","ArrayList","HashMap","Bundle","View","Context","Activity",
        "Fragment","Intent","File","TextView","Button","EditText","ImageView",
        "LinearLayout","RelativeLayout","RecyclerView","Adapter","ViewHolder",
        "FrameLayout","ConstraintLayout","ScrollView","Toast","Log","Handler"
    )

    private val pKw = Pattern.compile("\\b(${KEYWORDS.joinToString("|")})\\b")
    private val pTy = Pattern.compile("\\b(${TYPES.joinToString("|")})\\b")
    private val pStr = Pattern.compile("\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"")
    private val pChr = Pattern.compile("'[^'\\\\]*(\\\\.[^'\\\\]*)*'")
    private val pSlc = Pattern.compile("//[^\n]*")
    private val pMlc = Pattern.compile("/\\*[\\s\\S]*?\\*/")
    private val pNum = Pattern.compile("\\b\\d+(\\.\\d+)?[fFdDlL]?\\b")
    private val pAnn = Pattern.compile("@\\w+")
    private val pXtag = Pattern.compile("</?\\w[\\w:.-]*")
    private val pXattr = Pattern.compile("\\b[\\w:]+(?=\\s*=)")
    private val pXval = Pattern.compile("\"[^\"]*\"")
    private val pXcmt = Pattern.compile("<!--[\\s\\S]*?-->")

    enum class FileType { JAVA, KOTLIN, XML, UNKNOWN }

    fun detect(name: String): FileType = when (name.substringAfterLast('.').lowercase()) {
        "java" -> FileType.JAVA; "kt" -> FileType.KOTLIN; "xml" -> FileType.XML
        else -> FileType.UNKNOWN
    }

    fun highlight(s: Spannable, ft: FileType) {
        for (sp in s.getSpans(0, s.length, ForegroundColorSpan::class.java)) s.removeSpan(sp)
        val t = s.toString()
        when (ft) {
            FileType.JAVA, FileType.KOTLIN -> {
                apply(s, t, pNum, C_NUM); apply(s, t, pKw, C_KW); apply(s, t, pTy, C_TYPE)
                apply(s, t, pAnn, C_ANN); apply(s, t, pStr, C_STR); apply(s, t, pChr, C_STR)
                apply(s, t, pSlc, C_CMT); apply(s, t, pMlc, C_CMT)
            }
            FileType.XML -> {
                apply(s, t, pXtag, C_TAG); apply(s, t, pXattr, C_ATTR)
                apply(s, t, pXval, C_XVAL); apply(s, t, pXcmt, C_CMT)
            }
            else -> {}
        }
    }

    private fun apply(s: Spannable, t: String, p: Pattern, c: Int) {
        val m = p.matcher(t)
        while (m.find()) {
            s.setSpan(ForegroundColorSpan(c), m.start(), m.end(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
    }
}