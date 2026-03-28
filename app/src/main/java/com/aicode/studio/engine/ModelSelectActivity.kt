package com.aicode.studio.engine

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.*
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.aicode.studio.R

class ModelSelectActivity : AppCompatActivity() {

    private lateinit var mm      : ModelManager
    private lateinit var profile : HardwareAnalyzer.HardwareProfile
    private lateinit var allowed : Set<String>
    private lateinit var adapter : ModelCardAdapter
    private val pollHandler   = Handler(Looper.getMainLooper())
    private val pollRunnables = mutableMapOf<String, Runnable>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_select)

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "모델 선택"

        mm      = ModelManager(this)
        profile = HardwareAnalyzer.analyze(this)
        allowed = HardwareAnalyzer.allowedModels(profile)

        adapter = ModelCardAdapter(
            models   = InferenceConfig.ALL_MODELS,
            allowed  = allowed,
            mm       = mm,
            onAction = ::handleAction
        )

        val recycler = findViewById<RecyclerView>(R.id.recycler)
        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter        = adapter
        ViewCompat.setOnApplyWindowInsetsListener(recycler) { view, insets ->
            val navBar = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val pad = (8 * resources.displayMetrics.density).toInt()
            view.setPadding(pad, pad, pad, navBar.bottom + pad)
            insets
        }

        val tvHwSummary = findViewById<TextView>(R.id.tvHwSummary)
        tvHwSummary.text = buildString {
            append("백엔드: ${profile.recommendedBackend.displayName} (${profile.gpuRenderer})  ")
            append("RAM: ${profile.totalRamGb}GB  ")
            append("CPU: ${if (profile.hasDotProd) "DotProd" else "NEON"}${if (profile.hasI8mm) "+I8MM" else ""}")
        }
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }

    override fun onResume() {
        super.onResume()
        InferenceConfig.ALL_MODELS.forEach { model ->
            val info = mm.queryDownload(model) ?: return@forEach
            when {
                // 백그라운드에서 완료된 경우 → 설치 완료 처리
                info.isDone && !mm.isInstalled(model) -> {
                    mm.markInstalled(model)
                    adapter.refresh()
                    Toast.makeText(this, "✅ ${model.displayName} 설치 완료", Toast.LENGTH_SHORT).show()
                }
                // 아직 다운로드 중 → 폴링 재개
                info.isActive -> pollProgress(model)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pollHandler.removeCallbacksAndMessages(null)
    }

    enum class CardAction { INSTALL, ACTIVATE, DELETE, CANCEL }

    private fun handleAction(model: InferenceConfig.ModelDef, action: CardAction) {
        when (action) {
            CardAction.INSTALL   -> checkNetworkAndInstall(model)
            CardAction.ACTIVATE  -> activateModel(model)
            CardAction.CANCEL    -> {
                pollHandler.removeCallbacks(pollRunnables.remove(model.id) ?: return)
                mm.cancelDownload(model.id)
                adapter.clearProgress(model.id)
                adapter.refresh()
            }
            CardAction.DELETE    -> confirmDelete(model)
        }
    }

    private fun checkNetworkAndInstall(model: InferenceConfig.ModelDef) {
        val warn = NetworkChecker.cellularWarning(this, model.downloadSizeGb)
        if (warn != null) {
            AlertDialog.Builder(this)
                .setTitle("데이터 요금 주의")
                .setMessage(warn)
                .setPositiveButton("계속") { _, _ -> showInstallDialog(model) }
                .setNegativeButton("취소", null)
                .show()
        } else {
            showInstallDialog(model)
        }
    }

    private fun showInstallDialog(model: InferenceConfig.ModelDef) {
        val ramWarning = if (model.paramsBillion >= 7f)
            "\n⚠️ 대용량 모델입니다. 실행 전 다른 앱을 직접 종료해 RAM을 확보하세요.\n" else ""
        AlertDialog.Builder(this)
            .setTitle("모델 설치")
            .setMessage(
                "📦 ${model.displayName}\n\n" +
                "• 시리즈: ${model.series}\n" +
                "• 파라미터: ${model.paramsBillion}B\n" +
                "• 다운로드 크기: ${model.downloadSizeGb}GB\n" +
                "• 최소 RAM: ${model.minRamGb}GB\n" +
                "• Thinking: ${if (model.supportsThinking) "✅ 지원" else "❌ 미지원"}\n" +
                "• 양자화: Q4_K_M (4비트)\n" +
                ramWarning +
                "\n저장 위치:\nAndroid/data/com.aicode.studio/files/models/\n\n" +
                "앱 삭제 전까지 최초 1회만 다운로드합니다."
            )
            .setPositiveButton("설치") { _, _ -> startDownload(model) }
            .setNegativeButton("취소", null)
            .show()
    }

    private fun startDownload(model: InferenceConfig.ModelDef) {
        adapter.setProgress(model.id, 0f)
        mm.startSystemDownload(model)
        pollProgress(model)
    }

    private fun pollProgress(model: InferenceConfig.ModelDef) {
        if (pollRunnables.containsKey(model.id)) return  // 이미 폴링 중

        val runnable = object : Runnable {
            override fun run() {
                val info = mm.queryDownload(model)
                when {
                    info == null || info.isFailed -> {
                        pollRunnables.remove(model.id)
                        adapter.clearProgress(model.id)
                        adapter.refresh()
                        if (info?.isFailed == true) {
                            AlertDialog.Builder(this@ModelSelectActivity)
                                .setTitle("다운로드 실패")
                                .setMessage("네트워크 오류가 발생했습니다.\n다시 시도해주세요.")
                                .setPositiveButton("확인", null).show()
                        }
                    }
                    info.isDone -> {
                        pollRunnables.remove(model.id)
                        mm.markInstalled(model)
                        adapter.clearProgress(model.id)
                        adapter.refresh()
                        Toast.makeText(this@ModelSelectActivity, "✅ ${model.displayName} 설치 완료", Toast.LENGTH_LONG).show()
                        activateModel(model)
                    }
                    info.isActive -> {
                        adapter.setProgress(model.id, info.pct)
                        pollHandler.postDelayed(this, 1000)
                    }
                }
            }
        }
        pollRunnables[model.id] = runnable
        pollHandler.post(runnable)
    }

    private fun activateModel(model: InferenceConfig.ModelDef) {
        mm.activeModelId = model.id
        adapter.refresh()
        Toast.makeText(this, "✅ ${model.displayName} 활성화", Toast.LENGTH_SHORT).show()
        startForegroundService(Intent(this, AIInferenceService::class.java).apply {
            action = AIInferenceService.ACTION_START
            putExtra(InferenceConfig.KEY_MODEL_ID, model.id)
        })
    }

    private fun confirmDelete(model: InferenceConfig.ModelDef) {
        AlertDialog.Builder(this)
            .setTitle("모델 삭제")
            .setMessage("${model.displayName} (${model.downloadSizeGb}GB) 를 삭제합니다.")
            .setPositiveButton("삭제") { _, _ ->
                mm.deleteModel(model); adapter.refresh()
                Toast.makeText(this, "삭제 완료", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("취소", null).show()
    }

    class ModelCardAdapter(
        private val models  : List<InferenceConfig.ModelDef>,
        private val allowed : Set<String>,
        private val mm      : ModelManager,
        private val onAction: (InferenceConfig.ModelDef, CardAction) -> Unit
    ) : RecyclerView.Adapter<ModelCardAdapter.VH>() {

        private val progressMap = mutableMapOf<String, Float>()

        inner class VH(view: View) : RecyclerView.ViewHolder(view) {
            val tvModelName  : TextView    = view.findViewById(R.id.tvModelName)
            val tvModelMeta  : TextView    = view.findViewById(R.id.tvModelMeta)
            val tvModelStatus: TextView    = view.findViewById(R.id.tvModelStatus)
            val btnAction    : Button      = view.findViewById(R.id.btnAction)
            val progressBar  : ProgressBar = view.findViewById(R.id.progressBar)
        }

        override fun onCreateViewHolder(parent: ViewGroup, vt: Int): VH {
            val v = LayoutInflater.from(parent.context).inflate(R.layout.item_model_card, parent, false)
            return VH(v)
        }

        override fun getItemCount() = models.size

        override fun onBindViewHolder(holder: VH, pos: Int) {
            val m             = models[pos]
            val isAllowed     = m.id in allowed
            val isInstalled   = mm.isInstalled(m)
            val isActive      = mm.activeModelId == m.id
            val pct           = progressMap[m.id]
            val isDownloading = pct != null

            holder.tvModelName.text = m.displayName
            holder.tvModelMeta.text = buildString {
                append("${m.series}  ·  ${m.paramsBillion}B  ·  Q4_K_M  ·  ${m.downloadSizeGb}GB")
                append("\n최소 RAM: ${m.minRamGb}GB")
                if (m.supportsThinking) append("  ·  🧠 Thinking 지원")
            }
            holder.tvModelStatus.text = when {
                isActive      -> "✅ 사용 중"
                isInstalled   -> "⬛ 설치됨"
                isDownloading -> "⬇️ ${((pct ?: 0f) * 100).toInt()}%"
                !isAllowed    -> "⛔ RAM 부족 (${m.minRamGb}GB 필요)"
                else          -> "☐  미설치"
            }
            holder.tvModelStatus.setTextColor(
                holder.itemView.context.getColor(
                    when {
                        isActive   -> R.color.engine_accent_green
                        !isAllowed -> R.color.engine_accent_red
                        else       -> R.color.engine_text_secondary
                    }
                )
            )
            holder.progressBar.visibility = if (isDownloading) View.VISIBLE else View.GONE
            if (isDownloading) holder.progressBar.progress = ((pct ?: 0f) * 100).toInt()

            holder.btnAction.isEnabled = isAllowed
            holder.btnAction.alpha     = if (isAllowed) 1f else 0.4f
            holder.btnAction.text = when {
                isDownloading           -> "취소"
                isInstalled && isActive -> "사용 중"
                isInstalled             -> "활성화"
                else                    -> "설치"
            }
            holder.btnAction.backgroundTintList = android.content.res.ColorStateList.valueOf(
                holder.itemView.context.getColor(
                    when {
                        isDownloading -> R.color.engine_accent_red
                        isActive      -> R.color.engine_accent_green
                        else          -> R.color.engine_accent_blue
                    }
                )
            )
            holder.btnAction.setOnClickListener {
                when {
                    isDownloading            -> onAction(m, CardAction.CANCEL)
                    isInstalled && !isActive -> onAction(m, CardAction.ACTIVATE)
                    !isInstalled && isAllowed-> onAction(m, CardAction.INSTALL)
                }
            }
            holder.itemView.setOnLongClickListener {
                if (isInstalled) { onAction(m, CardAction.DELETE); true } else false
            }
        }

        fun setProgress(id: String, pct: Float) {
            progressMap[id] = pct
            notifyItemChanged(models.indexOfFirst { it.id == id }.takeIf { it >= 0 } ?: return)
        }
        fun clearProgress(id: String) {
            progressMap.remove(id)
            notifyItemChanged(models.indexOfFirst { it.id == id }.takeIf { it >= 0 } ?: return)
        }
        fun refresh() { notifyDataSetChanged() }
    }
}
