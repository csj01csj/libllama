package com.aicode.studio

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.Editable
import android.text.TextWatcher
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.drawerlayout.widget.DrawerLayout
import com.aicode.studio.ai.AIAgentManager
import com.aicode.studio.ai.LocalAIManager
import com.aicode.studio.build.BuildManager
import com.aicode.studio.editor.AutoCompleteManager
import com.aicode.studio.editor.CodeEditorView
import com.aicode.studio.editor.SymbolBarManager
import com.aicode.studio.editor.TabManager
import com.aicode.studio.explorer.FileExplorerAdapter
import com.aicode.studio.project.ProjectManager
import com.aicode.studio.util.LogManager
import com.aicode.studio.util.PrefsManager
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var drawer: DrawerLayout
    private lateinit var codeEditor: CodeEditorView
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var fileListView: ListView
    private lateinit var tvProject: TextView
    private lateinit var tvCurrentFile: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvCursorPos: TextView
    private lateinit var etAiInput: EditText
    private lateinit var btnSendAI: ImageButton
    private lateinit var btnStopAI: ImageButton
    private lateinit var btnRun: Button
    private lateinit var btnSync: ImageButton
    private lateinit var btnSwitchProject: Button
    private lateinit var tabContainer: LinearLayout
    private lateinit var tabScrollView: HorizontalScrollView
    private lateinit var symbolBar: LinearLayout
    private lateinit var logPanel: LinearLayout
    private lateinit var apiConfigRow: LinearLayout
    private lateinit var apiListContainer: LinearLayout
    private lateinit var btnAddApi: Button
    private lateinit var switchLocalAI: Switch
    private lateinit var btnManageLocalAI: Button

    // ─── Managers ─────────────────────────────────
    private lateinit var logger: LogManager
    private lateinit var projectMgr: ProjectManager
    private lateinit var explorerAdapter: FileExplorerAdapter
    private lateinit var aiAgent: AIAgentManager
    private lateinit var localAIMgr: LocalAIManager
    private lateinit var buildMgr: BuildManager
    private lateinit var tabMgr: TabManager
    private lateinit var symbolBarMgr: SymbolBarManager
    private lateinit var autoComplete: AutoCompleteManager

    private var currentProject: File? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var buildToolsReady = false
    private var logExpanded = true

    private val apiConfigsList = mutableListOf<AIAgentManager.ApiConfig>()

    data class PendingKeystore(val pass: String, val alias: String, val dname: String)
    private var pendingKeystore: PendingKeystore? = null

    // ─── Launcher ─────────────────────────────────
    private val pickKeystoreLauncher = registerForActivityResult(androidx.activity.result.contract.ActivityResultContracts.GetContent()) { uri ->
        uri?.let { handleKeystorePicked(it) }
    }

    private val createKeystoreLauncher = registerForActivityResult(androidx.activity.result.contract.ActivityResultContracts.CreateDocument()) { uri ->
        uri?.let { handleKeystoreCreated(it) }
    }

    // ─── Lifecycle ────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        setTheme(R.style.Theme_AICode)   // force-apply before super so AppCompat sees it
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 상태바 숨기기
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.statusBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        // 하단 네비게이션바 패딩 적용
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.drawerLayout)) { _, insets ->
            val navBar = insets.getInsets(WindowInsetsCompat.Type.navigationBars())
            findViewById<View>(R.id.aiPanel).setPadding(0, 0, 0, navBar.bottom)
            insets
        }

        createNotificationChannels()

        bindViews()    // 1. 뷰 바인딩
        initManagers() // 2. 매니저 초기화
        loadApiConfigs() // 3. 설정 로드
        
        setupListeners()
        loadSavedState()
        checkPermissions()
        buildMgr.prepareTools {
            buildToolsReady = true
            mainHandler.post { logger.logSystem("Build tools ready") }
        }
        window.decorView.post { showProjectSelector() }
    }

    private fun createNotificationChannels() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val nm = getSystemService(android.app.NotificationManager::class.java)
            nm.createNotificationChannel(
                android.app.NotificationChannel(
                    "ai_inference", "AI Inference", android.app.NotificationManager.IMPORTANCE_LOW
                ).apply { description = "On-device AI inference service" }
            )
            nm.createNotificationChannel(
                android.app.NotificationChannel(
                    "model_download", "Model Download", android.app.NotificationManager.IMPORTANCE_LOW
                ).apply { description = "GGUF model download progress" }
            )
        }
    }

    private fun bindViews() {
        drawer = findViewById(R.id.drawerLayout)
        codeEditor = findViewById(R.id.codeEditor)
        logText = findViewById(R.id.logText)
        logScroll = findViewById(R.id.logScroll)
        fileListView = findViewById(R.id.fileListView)
        tvProject = findViewById(R.id.tvProjectTitle)
        tvCurrentFile = findViewById(R.id.tvCurrentFile)
        tvStatus = findViewById(R.id.tvStatus)
        tvCursorPos = findViewById(R.id.tvCursorPos)
        etAiInput = findViewById(R.id.etAiInput)
        btnSendAI = findViewById(R.id.btnSendAI)
        btnStopAI = findViewById(R.id.btnStopAI)
        btnRun = findViewById(R.id.btnRun)
        tabContainer = findViewById(R.id.tabContainer)
        tabScrollView = findViewById(R.id.tabScrollView)
        symbolBar = findViewById(R.id.symbolBar)
        logPanel = findViewById(R.id.logPanel)
        apiConfigRow = findViewById(R.id.apiConfigRow)
        apiListContainer = findViewById(R.id.apiListContainer)
        btnAddApi = findViewById(R.id.btnAddApi)
        switchLocalAI = findViewById(R.id.switchLocalAI)
        btnManageLocalAI = findViewById(R.id.btnManageLocalAI)
        btnSync = findViewById(R.id.btnSync)
        btnSwitchProject = findViewById(R.id.btnSwitchProject)

        
        // Force selectable
        logText.setTextIsSelectable(true)
        logText.isFocusable = true
        logText.isFocusableInTouchMode = true
    }

    private fun checkPermissions() {
        // 알림 권한 (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                    arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 101)
            }
        }
        // 배터리 최적화 제외 요청 (백그라운드 AI 서비스 유지)
        val pm = getSystemService(android.os.PowerManager::class.java)
        if (!pm.isIgnoringBatteryOptimizations(packageName)) {
            AlertDialog.Builder(this)
                .setTitle("배터리 최적화")
                .setMessage("백그라운드 AI 서비스가 종료되지 않도록 배터리 최적화 예외를 설정해주세요.")
                .setPositiveButton("설정") { _, _ ->
                    startActivity(android.content.Intent(
                        Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                        data = Uri.parse("package:$packageName")
                    })
                }
                .setNegativeButton("나중에", null)
                .show()
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                AlertDialog.Builder(this)
                    .setTitle("권한 필요")
                    .setMessage("안드로이드 11 이상에서는 빌드 및 파일 관리를 위해 '모든 파일 접근 권한'이 필요합니다.")
                    .setPositiveButton("설정으로 이동") { _, _ ->
                        try {
                            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                            intent.data = Uri.parse("package:$packageName")
                            startActivity(intent)
                        } catch (e: Exception) {
                            val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                            startActivity(intent)
                        }
                    }
                    .setNegativeButton("취소", null)
                    .show()
            }
        } else {
            val perms = arrayOf(
                android.Manifest.permission.READ_EXTERNAL_STORAGE,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            val needed = perms.filter {
                ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
            }
            if (needed.isNotEmpty()) {
                ActivityCompat.requestPermissions(this, needed.toTypedArray(), 100)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (codeEditor.hasFile()) codeEditor.saveFile()
        currentProject?.let { PrefsManager.saveLastProject(this, it.absolutePath) }
    }

    override fun onStop() {
        super.onStop()
        if (codeEditor.hasFile()) codeEditor.saveFile()
    }

    // ─── Init ─────────────────────────────────────

    private fun loadApiConfigs() {
        val saved = PrefsManager.getApiConfigs(this)
        apiConfigsList.clear()
        try {
            val arr = org.json.JSONArray(saved)
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                apiConfigsList.add(AIAgentManager.ApiConfig(o.getString("key"), o.getString("model")))
            }
        } catch (_: Exception) {}
        
        if (apiConfigsList.isEmpty()) apiConfigsList.add(AIAgentManager.ApiConfig("", "..."))
        
        // Deduplicate: Keep first occurrence
        val unique = mutableListOf<AIAgentManager.ApiConfig>()
        val keysSeen = mutableSetOf<String>()
        apiConfigsList.forEach {
            if (it.key.isEmpty() || keysSeen.add(it.key)) {
                unique.add(it)
            }
        }
        apiConfigsList.clear()
        apiConfigsList.addAll(unique.take(7)) // Max 7
        
        refreshApiUI()
    }

    private fun refreshApiUI() {
        apiListContainer.removeAllViews()
        apiConfigsList.forEachIndexed { index, config ->
            addApiView(index, config)
        }
        aiAgent.setApiConfigs(apiConfigsList.filter { it.key.isNotEmpty() && it.model != "..." })
    }

    private fun addApiView(index: Int, config: AIAgentManager.ApiConfig) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 4, 0, 4) }
        }

        val etKey = EditText(this).apply {
            hint = "API Key ${index + 1}"
            setText(config.key)
            textSize = 11f
            setTextColor(Color.WHITE)
            setHintTextColor(Color.parseColor("#555555"))
            setBackgroundColor(Color.parseColor("#2D2D2D"))
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            layoutParams = LinearLayout.LayoutParams(0, 100, 1f)
            setPadding(16, 0, 16, 0)
        }

        val spModel = Spinner(this).apply {
            layoutParams = LinearLayout.LayoutParams(450, 100).apply { marginStart = 8 }
            setBackgroundColor(Color.parseColor("#2D2D2D"))
        }

        val models = mutableListOf(config.model)
        if (models[0].isEmpty()) models[0] = "..."
        
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        spModel.adapter = adapter

        // Duplicate Check
        val isDuplicate = index > 0 && apiConfigsList.take(index).any { it.key.isNotEmpty() && it.key == config.key }
        if (isDuplicate) etKey.setTextColor(Color.RED)

        etKey.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val newKey = s.toString().trim()
                val oldKey = apiConfigsList[index].key
                apiConfigsList[index] = apiConfigsList[index].copy(key = newKey)
                
                if (newKey.length < 10) {
                    adapter.clear(); adapter.add("..."); adapter.notifyDataSetChanged()
                    saveConfigs()
                    return
                }

                if (newKey != oldKey) {
                    fetchModels(newKey, spModel, adapter, index)
                }
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })

        spModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, pos: Int, p3: Long) {
                val selected = adapter.getItem(pos) ?: "..."
                apiConfigsList[index] = apiConfigsList[index].copy(model = selected)
                saveConfigs()
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        if (config.key.length >= 10) {
            fetchModels(config.key, spModel, adapter, index, config.model)
        }

        row.addView(etKey)
        row.addView(spModel)
        
        if (index > 0) {
            val btnDel = ImageButton(this).apply {
                setImageResource(android.R.drawable.ic_menu_delete)
                setBackgroundResource(0)
                setColorFilter(Color.GRAY)
                setOnClickListener {
                    apiConfigsList.removeAt(index)
                    saveConfigs()
                    refreshApiUI()
                }
            }
            row.addView(btnDel)
        }

        apiListContainer.addView(row)
    }

    private fun fetchModels(key: String, sp: Spinner, adapter: ArrayAdapter<String>, index: Int, current: String? = null) {
        aiAgent.fetchModels(key) { list ->
            mainHandler.post {
                adapter.clear()
                adapter.addAll(list)
                val target = current ?: apiConfigsList[index].model
                val pos = list.indexOf(target).coerceAtLeast(0)
                val selected = list.getOrElse(pos) { list.firstOrNull() ?: "gemini-2.0-flash" }
                apiConfigsList[index] = apiConfigsList[index].copy(model = selected)
                sp.setSelection(pos)
                adapter.notifyDataSetChanged()
                saveConfigs()
            }
        }
    }

    private fun saveConfigs() {
        val arr = org.json.JSONArray()
        apiConfigsList.forEach {
            arr.put(org.json.JSONObject().put("key", it.key).put("model", it.model))
        }
        PrefsManager.saveApiConfigs(this, arr.toString())
        aiAgent.setApiConfigs(apiConfigsList.filter { it.key.isNotEmpty() && it.model != "..." })
    }

    private fun initManagers() {
        logger = LogManager(logText, logScroll)
        projectMgr = ProjectManager(this)
        buildMgr = BuildManager(this, logger)
        aiAgent = AIAgentManager(logger)
        localAIMgr = LocalAIManager(this, logger)
        autoComplete = AutoCompleteManager(this, codeEditor)

        explorerAdapter = FileExplorerAdapter(
            onFileClick = { file -> openFileInEditor(file) },
            onFolderToggle = { refreshExplorer() },
            onLongClick = { file, _ -> showFileContextMenu(file); true }
        )
        fileListView.adapter = explorerAdapter

        tabMgr = TabManager(tabContainer, tabScrollView,
            onTabSelect = { file -> openFileInEditor(file) },
            onTabClose = { }
        )

        symbolBarMgr = SymbolBarManager(symbolBar, codeEditor)
        symbolBarMgr.setup()

        aiAgent.setCallback(object : AIAgentManager.Callback {
            override fun onThought(thought: String) {
                mainHandler.post {
                    tvStatus.text = "AI: 답변 생성 중..."
                    logger.logAI("($thought)")
                    if (!logExpanded) toggleLogPanel()
                }
            }

            override fun onFileUpdated(path: String, success: Boolean, msg: String) {
                mainHandler.post {
                    refreshExplorer()
                    val curFile = codeEditor.getFile()
                    if (curFile != null && curFile.absolutePath.endsWith(path) && curFile.exists()) {
                        codeEditor.setExternal(curFile.readText())
                    }
                }
            }

            override fun onDeleteRequested(path: String, onConfirm: (Boolean) -> Unit) {
                mainHandler.post {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("AI 삭제 요청")
                        .setMessage("삭제: $path")
                        .setPositiveButton("허용") { _, _ -> onConfirm(true) }
                        .setNegativeButton("거절") { _, _ -> onConfirm(false) }
                        .setCancelable(false).show()
                }
            }

            override fun onCompleted(summary: String) {
                mainHandler.post {
                    tvStatus.text = "AI 완료"
                    tvStatus.setTextColor(Color.parseColor("#4CAF50"))
                    btnSendAI.visibility = View.VISIBLE
                    btnStopAI.visibility = View.GONE
                    btnSendAI.isEnabled = true
                    etAiInput.text.clear()
                    refreshExplorer()
                    
                    if (summary.isNotEmpty()) {
                        logger.logAI(summary)
                        if (!logExpanded) toggleLogPanel()
                    }
                }
            }

            override fun onError(error: String) {
                mainHandler.post {
                    tvStatus.text = "오류: ${error.take(60)}"
                    tvStatus.setTextColor(Color.parseColor("#FF5555"))
                    btnSendAI.visibility = View.VISIBLE
                    btnStopAI.visibility = View.GONE
                    btnSendAI.isEnabled = true
                }
            }

            override fun onStatusChanged(status: String) {
                mainHandler.post {
                    tvStatus.text = status
                    tvStatus.setTextColor(Color.parseColor("#FFA500"))
                }
            }
        })
    }

    private fun setupListeners() {
        findViewById<ImageButton>(R.id.btnMenu).setOnClickListener {
            drawer.openDrawer(GravityCompat.START)
        }

        btnRun.setOnClickListener { runBuild() }

        findViewById<ImageButton>(R.id.btnUndo).setOnClickListener {
            codeEditor.onKeyDown(
                KeyEvent.KEYCODE_Z,
                KeyEvent(0, 0, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_Z, 0, KeyEvent.META_CTRL_ON)
            )
        }
        findViewById<ImageButton>(R.id.btnRedo).setOnClickListener {
            codeEditor.onKeyDown(
                KeyEvent.KEYCODE_Y,
                KeyEvent(0, 0, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_Y, 0, KeyEvent.META_CTRL_ON)
            )
        }

        btnSendAI.setOnClickListener { sendAICommand() }
        btnStopAI.setOnClickListener {
            aiAgent.stop()
            btnSendAI.visibility = View.VISIBLE
            btnStopAI.visibility = View.GONE
        }

        findViewById<ImageButton>(R.id.btnToggleApiConfig).setOnClickListener {
            apiConfigRow.visibility =
                if (apiConfigRow.visibility == View.GONE) View.VISIBLE else View.GONE
            if (apiConfigRow.visibility == View.GONE) {
                loadApiConfigs() // Deduplicate & enforce limit on close
            }
        }

        btnAddApi.setOnClickListener {
            if (apiConfigsList.size < 7) {
                apiConfigsList.add(AIAgentManager.ApiConfig("", "..."))
                refreshApiUI()
            } else {
                Toast.makeText(this, "최대 7개까지 가능합니다.", Toast.LENGTH_SHORT).show()
            }
        }

        switchLocalAI.setOnCheckedChangeListener { _, isChecked ->
            applyLocalAIMode(isChecked)
        }

        btnManageLocalAI.setOnClickListener {
            localAIMgr.openManageActivity()
        }

        findViewById<ImageButton>(R.id.btnToggleLog).setOnClickListener { toggleLogPanel() }
        findViewById<ImageButton>(R.id.btnClearLog).setOnClickListener { logger.clear() }

        // Drawer Buttons
        findViewById<Button>(R.id.btnNewProject).setOnClickListener { showCreateProjectWizard() }
        btnSync.setOnClickListener { syncProject() }
        btnSwitchProject.setOnClickListener { showProjectSelector() }


        codeEditor.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (codeEditor.isBusy) return // 로딩 중이거나 작업 중이면 중단
                codeEditor.getFile()?.let { 
                    tabMgr.markModified(it, true)
                    codeEditor.saveFile() 
                }
            }
        })

        codeEditor.onCursorMoved = { l, c ->
            tvCursorPos.text = "L:$l C:$c"
        }
    }

    private fun loadSavedState() {
        val last = PrefsManager.getLastProject(this)
        if (last.isNotEmpty()) {
            val f = File(last)
            if (f.exists()) loadProject(f)
        }
        val fontSize = PrefsManager.getFontSize(this)
        codeEditor.setTextSize(fontSize)

        // Local AI 모드 복원 (isChecked 설정이 리스너를 통해 applyLocalAIMode 호출)
        val localMode = PrefsManager.getLocalAIMode(this)
        switchLocalAI.isChecked = localMode
    }

    private fun applyLocalAIMode(enabled: Boolean) {
        PrefsManager.saveLocalAIMode(this, enabled)
        if (enabled) {
            localAIMgr.connect()
            // 서비스가 죽어서 재생성된 경우를 대비해 저장된 모델로 ACTION_START 전송
            val savedModelId = com.aicode.studio.engine.ModelManager(this).activeModelId
            if (savedModelId.isNotEmpty()) {
                startForegroundService(
                    android.content.Intent(this, com.aicode.studio.engine.AIInferenceService::class.java).apply {
                        action = com.aicode.studio.engine.AIInferenceService.ACTION_START
                        putExtra(com.aicode.studio.engine.InferenceConfig.KEY_MODEL_ID, savedModelId)
                    }
                )
            }
            aiAgent.setLocalAI(localAIMgr, true)
            switchLocalAI.setTextColor(android.graphics.Color.parseColor("#4CAF50"))
            logger.logSystem("Local AI 모드 활성화")
        } else {
            aiAgent.setLocalAI(null, false)
            localAIMgr.stopAndDisconnect()
            switchLocalAI.setTextColor(android.graphics.Color.parseColor("#AAAAAA"))
            logger.logSystem("Local AI 모드 비활성화")
        }
    }

    // ─── Project Management ───────────────────────

    private fun showProjectSelector() {
        val projects = projectMgr.listProjects()
        if (projects.isEmpty()) {
            showCreateProjectWizard()
            return
        }
        val names = projects.map { it.name }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("프로젝트 선택")
            .setItems(names) { _, which -> loadProject(projects[which]) }
            .setPositiveButton("+ 새 프로젝트") { _, _ -> showCreateProjectWizard() }
            .setNeutralButton("삭제") { _, _ -> showDeleteProjectDialog(projects) }
            .show()
    }

    private fun showDeleteProjectDialog(projects: List<File>) {
        val names = projects.map { it.name }.toTypedArray()
        val selected = mutableSetOf<Int>()
        AlertDialog.Builder(this)
            .setTitle("삭제할 프로젝트")
            .setMultiChoiceItems(names, null) { _, which, checked ->
                if (checked) selected.add(which) else selected.remove(which)
            }
            .setPositiveButton("삭제") { _, _ ->
                selected.forEach { projectMgr.deleteProject(projects[it]) }
                showProjectSelector()
            }
            .setNegativeButton("취소", null).show()
    }

    private fun showCreateProjectWizard() {
        val view = layoutInflater.inflate(R.layout.dialog_project_wizard, null)
        val etName = view.findViewById<EditText>(R.id.etName)
        val etPkg = view.findViewById<EditText>(R.id.etPackage)
        val etMinSdk = view.findViewById<EditText>(R.id.etMinSdk)
        val spLang = view.findViewById<Spinner>(R.id.spLanguage)
        val spTmpl = view.findViewById<Spinner>(R.id.spTemplate)

        spTmpl.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            ProjectManager.Template.values().map { it.label }
        )

        spTmpl.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, pos: Int, p3: Long) {
                val tmpl = ProjectManager.Template.values()[pos]
                val langs = ProjectManager.Language.values().filter { lang ->
                    if (tmpl == ProjectManager.Template.COMPOSE) lang == ProjectManager.Language.KOTLIN
                    else true
                }
                spLang.adapter = ArrayAdapter(
                    this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                    langs.map { it.label }
                )
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        etName.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val name = s.toString().lowercase().replace(Regex("[^a-z0-9]"), "")
                if (name.isNotEmpty()) etPkg.setText("com.example.$name")
            }
        })

        AlertDialog.Builder(this).setView(view)
            .setPositiveButton("생성") { _, _ ->
                val name = etName.text.toString().trim()
                val pkg = etPkg.text.toString().trim()
                val minSdk = etMinSdk.text.toString().toIntOrNull() ?: 24
                
                if (name.isEmpty() || pkg.isEmpty()) {
                    Toast.makeText(this, "이름과 패키지명을 입력하세요", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                
                if (projectMgr.isProjectExists(name)) {
                    Toast.makeText(this, "이미 존재하는 프로젝트 이름입니다.", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                
                if (projectMgr.isPackageConflict(pkg)) {
                    Toast.makeText(this, "다른 프로젝트에서 이미 사용 중인 패키지명입니다.", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }

                val tmpl = ProjectManager.Template.values()[spTmpl.selectedItemPosition]
                val selectedLangLabel = spLang.selectedItem?.toString() ?: "Java"
                val lang = ProjectManager.Language.values().find { it.label == selectedLangLabel } ?: ProjectManager.Language.JAVA
                val cfg = ProjectManager.ProjectConfig(name, pkg, minSdk, language = lang, template = tmpl)
                val root = try {
                    projectMgr.createProject(cfg)
                } catch (e: Exception) {
                    Toast.makeText(this, e.message, Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                loadProject(root)
                logger.logSystem("프로젝트 생성: $name ($pkg)")
            }
            .setNegativeButton("취소", null).show()
    }

    private fun syncProject() {
        val project = currentProject ?: run { Toast.makeText(this, "프로젝트를 선택하세요", Toast.LENGTH_SHORT).show(); return }
        
        logger.logSystem("Gradle Sync 시작: ${project.name}")
        if (!logExpanded) toggleLogPanel()

        Thread {
            try {
                val depsDir = File(cacheDir, "deps").apply { if (!exists()) mkdirs() }
                val buildGradle = File(project, "app/build.gradle")
                val buildGradleKts = File(project, "app/build.gradle.kts")
                val gradleFile = if (buildGradleKts.exists()) buildGradleKts else buildGradle
                
                if (!gradleFile.exists()) {
                    mainHandler.post { logger.logError("빌드 스크립트(build.gradle 또는 .kts)를 찾을 수 없습니다.") }
                    return@Thread
                }

                val content = gradleFile.readText()
                val depsList = mutableListOf<String>()
                
                // 정규표현식 개선: implementation, api, kapt, annotationProcessor 등 지원
                // 1. 기본 문자열 형식 추출: implementation "group:name:version" or implementation("group:name:version")
                val configurationKeywords = listOf("implementation", "api", "kapt", "annotationProcessor", "compileOnly", "runtimeOnly")
                configurationKeywords.forEach { kw ->
                    Regex("$kw\\s*\\(?\\s*['\"]([^'\"]+)['\"]\\s*\\)?").findAll(content).forEach { 
                        val dep = it.groupValues[1]
                        if (dep.contains(":")) depsList.add(dep)
                    }
                    
                    // KTS named parameters: implementation(group = "...", name = "...", version = "...")
                    Regex("$kw\\s*\\(([^)]+)\\)").findAll(content).forEach { 
                        val inner = it.groupValues[1]
                        val g = Regex("group\\s*=\\s*['\"]([^'\"]+)['\"]").find(inner)?.groupValues?.get(1)
                        val n = Regex("name\\s*=\\s*['\"]([^'\"]+)['\"]").find(inner)?.groupValues?.get(1)
                        val v = Regex("version\\s*=\\s*['\"]([^'\"]+)['\"]").find(inner)?.groupValues?.get(1)
                        if (g != null && n != null && v != null) {
                            depsList.add("$g:$n:$v")
                        }
                    }
                }
                
                // Kotlin 프로젝트인 경우 stdlib 강제 추가
                val config = projectMgr.getConfig(project)
                if (config?.language == ProjectManager.Language.KOTLIN) {
                    if (!depsList.any { it.contains("kotlin-stdlib") }) {
                        depsList.add("org.jetbrains.kotlin:kotlin-stdlib:1.9.0")
                    }
                }
                
                // Lottie 감지 시 AndroidX 코어 의존성 자동 보완
                if (depsList.any { it.contains("lottie") }) {
                    val coreDeps = listOf(
                        "androidx.appcompat:appcompat:1.6.1",
                        "androidx.activity:activity:1.6.0",
                        "androidx.activity:activity-ktx:1.6.0",
                        "androidx.fragment:fragment:1.3.6",
                        "androidx.fragment:fragment-ktx:1.3.6",
                        "androidx.core:core:1.9.0",
                        "androidx.core:core-ktx:1.9.0",
                        "androidx.vectordrawable:vectordrawable:1.1.0",
                        "androidx.vectordrawable:vectordrawable-animated:1.1.0",
                        "androidx.tracing:tracing:1.1.0",
                        "androidx.annotation:annotation:1.5.0",
                        "androidx.annotation:annotation-experimental:1.3.0",
                        "androidx.lifecycle:lifecycle-runtime:2.5.1",
                        "androidx.lifecycle:lifecycle-common:2.5.1",
                        "androidx.lifecycle:lifecycle-viewmodel:2.5.1",
                        "androidx.lifecycle:lifecycle-viewmodel-ktx:2.5.1",
                        "androidx.arch.core:core-common:2.1.0",
                        "androidx.collection:collection:1.2.0",
                        "androidx.resourceinspection:resourceinspection-annotation:1.0.1"
                    )
                    coreDeps.forEach { cd ->
                        val art = cd.split(":")[1]
                        if (!depsList.any { it.contains(art) }) {
                            depsList.add(cd)
                            mainHandler.post { logger.logSystem("의존성 보완: $art") }
                        }
                    }
                }

                val okClient = okhttp3.OkHttpClient()
                val resolver = com.tyron.resolver.TransitiveResolver(okClient)
                resolver.onProgress = { msg ->
                    mainHandler.post { logger.logSystem(msg) }
                }
                
                mainHandler.post { logger.logSystem("의존성 트리 분석 중...") }
                
                val rootDeps = depsList.mapNotNull { 
                    val p = it.split(":")
                    if (p.size == 3) com.tyron.resolver.TransitiveResolver.Dep(p[0], p[1], p[2]) else null
                }
                
                val allDeps = resolver.resolve(rootDeps)
                mainHandler.post { logger.logSystem("분석 완료: 총 ${allDeps.size}개 라이브러리 필요") }

                val resolvedDepsFile = File(project, "app/resolved_deps.txt")
                val resolvedPaths = mutableListOf<String>()

                allDeps.forEach { dep ->
                    val aarName = "${dep.a}-${dep.v}.aar"
                    val jarName = "${dep.a}-${dep.v}.jar"
                    val localAar = File(depsDir, aarName)
                    val localJar = File(depsDir, jarName)

                    val alreadyExists = localAar.exists() || localJar.exists()

                    if (!alreadyExists) {
                        mainHandler.post { logger.logSystem("의존성 다운로드: $dep") }
                        
                        var downloaded = false
                        val repos = listOf(
                            "https://maven.google.com/",
                            "https://repo1.maven.org/maven2/",
                            "https://jcenter.bintray.com/"
                        )
                        
                        for (base in repos) {
                            if (tryDownload(okClient, base + dep.toPath("aar"), localAar)) {
                                mainHandler.post { logger.logSystem("  - [성공] AAR") }
                                downloaded = true; break
                            }
                            if (tryDownload(okClient, base + dep.toPath("jar"), localJar)) {
                                mainHandler.post { logger.logSystem("  - [성공] JAR") }
                                downloaded = true; break
                            }
                        }
                        if (!downloaded) mainHandler.post { logger.logError("  - [실패] $dep") }
                        else {
                            if (localAar.exists()) resolvedPaths.add(localAar.absolutePath)
                            else if (localJar.exists()) resolvedPaths.add(localJar.absolutePath)
                        }
                    } else {
                        // 이미 존재하면 로그 없이 경로만 리스트에 추가
                        if (localAar.exists()) resolvedPaths.add(localAar.absolutePath)
                        else if (localJar.exists()) resolvedPaths.add(localJar.absolutePath)
                    }
                }
                
                // 최종 분석된 경로들 저장
                resolvedDepsFile.writeText(resolvedPaths.joinToString("\n"))

                mainHandler.post {
                    refreshExplorer()
                    logger.logSystem("Gradle Sync 완료!")
                    Toast.makeText(this, "Sync 완료", Toast.LENGTH_SHORT).show()
                    drawer.closeDrawers()
                }
            } catch (e: Exception) {
                mainHandler.post { logger.logError("Sync 중 치명적 오류: ${e.message}") }
            }
        }.start()
    }

    private fun tryDownload(client: okhttp3.OkHttpClient, url: String, target: File): Boolean {
        return try {
            val req = okhttp3.Request.Builder().url(url).build()
            client.newCall(req).execute().use { resp ->
                if (resp.isSuccessful) {
                    resp.body?.byteStream()?.use { input ->
                        target.outputStream().use { output -> input.copyTo(output) }
                    }
                    true
                } else false
            }
        } catch (_: Exception) { false }
    }

    private fun loadProject(root: File) {
        currentProject?.let { oldRoot ->
            codeEditor.saveFile()
            val openFiles = tabMgr.getOpenFiles().map { it.absolutePath }
            val activePath = tabMgr.getActiveFile()?.absolutePath ?: ""
            val json = org.json.JSONObject().apply {
                put("openFiles", org.json.JSONArray(openFiles))
                put("activeFile", activePath)
            }
            PrefsManager.saveProjectTabs(this, oldRoot.name, json.toString())
        }

        currentProject = root
        tvProject.text = root.name
        aiAgent.loadProject(root)
        
        if (::tabMgr.isInitialized) {
            tabMgr.closeAllTabs()
        }
        codeEditor.clearEditor()
        refreshExplorer()

        val saved = PrefsManager.getProjectTabs(this, root.name)
        if (saved != "[]") {
            try {
                val json = org.json.JSONObject(saved)
                val files = json.getJSONArray("openFiles")
                val active = json.getString("activeFile")
                
                for (i in 0 until files.length()) {
                    val f = File(files.getString(i))
                    if (f.exists()) tabMgr.openTab(f)
                }
                
                if (active.isNotEmpty()) {
                    val af = File(active)
                    if (af.exists()) openFileInEditor(af)
                }
            } catch (_: Exception) {}
        }

        if (tabMgr.getActiveFile() == null) {
            val config = projectMgr.getConfig(root)
            if (config != null) {
                val pp = config.packageName.replace(".", "/")
                val ext = if (config.language == ProjectManager.Language.KOTLIN) "kt" else "java"
                val main = File(root, "app/src/main/java/$pp/MainActivity.$ext")
                if (main.exists()) openFileInEditor(main)
            }
        }

        drawer.closeDrawers()
        PrefsManager.saveLastProject(this, root.absolutePath)
    }

    // ─── File Explorer ────────────────────────────

    private fun refreshExplorer() {
        currentProject?.let { explorerAdapter.setRoot(it) }
    }

    private fun openFileInEditor(file: File) {
        if (!file.exists()) return
        codeEditor.saveFile()
        
        try {
            codeEditor.openFile(file)
            tabMgr.openTab(file)
            tvCurrentFile.text = file.name
        } catch (e: Exception) {
            Toast.makeText(this, "파일 열기 실패: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showFileContextMenu(file: File) {
        val items = if (file.isDirectory) arrayOf("새 파일", "새 폴더", "삭제") else arrayOf("삭제", "이름 변경")
        AlertDialog.Builder(this).setTitle(file.name).setItems(items) { _, which ->
            when (items[which]) {
                "새 파일" -> showCreateFileDialog(file, false)
                "새 폴더" -> showCreateFileDialog(file, true)
                "삭제" -> {
                    file.deleteRecursively()
                    refreshExplorer()
                }
                "이름 변경" -> { /* TODO */ }
            }
        }.show()
    }

    private fun showCreateFileDialog(parent: File, isDir: Boolean) {
        val et = EditText(this)
        AlertDialog.Builder(this).setTitle(if (isDir) "새 폴더" else "새 파일").setView(et)
            .setPositiveButton("생성") { _, _ ->
                val name = et.text.toString()
                if (name.isNotEmpty()) {
                    val f = File(parent, name)
                    if (isDir) f.mkdirs() else f.createNewFile()
                    refreshExplorer()
                }
            }.show()
    }

    // ─── Build ────────────────────────────────────

    private fun runBuild() {
        val project = currentProject ?: run { Toast.makeText(this, "프로젝트를 선택하세요", Toast.LENGTH_SHORT).show(); return }
        if (!buildToolsReady) { Toast.makeText(this, "빌드 도구 준비 중...", Toast.LENGTH_SHORT).show(); return }

        val items = arrayOf("Debug Build (내장 키 사용)", "Release Build (커스텀 키스토어)")
        AlertDialog.Builder(this)
            .setTitle("빌드 방식 선택")
            .setItems(items) { _, which ->
                if (which == 0) executeBuild(project, null)
                else showReleaseBuildFlow(project)
            }.show()
    }

    private fun showReleaseBuildFlow(project: File) {
        val options = arrayOf("기존 키스토어 파일 선택", "새 키스토어 생성")
        AlertDialog.Builder(this)
            .setTitle("서명 설정")
            .setItems(options) { _, which ->
                if (which == 0) selectKeystoreFile()
                else createKeystoreFlow()
            }.show()
    }

    private fun selectKeystoreFile() {
        pickKeystoreLauncher.launch("*/*")
    }

    private fun createKeystoreFlow() {
        showKeystoreDetailsDialog()
    }

    private fun handleKeystorePicked(uri: Uri) {
        val path = getPathFromUri(uri) ?: uri.toString()
        requestPasswordAndBuild(path)
    }

    private fun handleKeystoreCreated(uri: Uri) {
        val info = pendingKeystore ?: return
        val path = getPathFromUri(uri) ?: return
        val file = File(path)

        buildMgr.createKeystore(file, info.pass, info.alias, info.dname) { success, msg ->
            mainHandler.post {
                if (success) {
                    Toast.makeText(this, "키스토어가 생성되었습니다.", Toast.LENGTH_SHORT).show()
                    requestPasswordAndBuild(file.absolutePath)
                } else {
                    AlertDialog.Builder(this).setTitle("생성 실패").setMessage(msg).setPositiveButton("확인", null).show()
                }
            }
        }
        pendingKeystore = null
    }

    private fun requestPasswordAndBuild(path: String) {
        val project = currentProject ?: return
        val savedPass = PrefsManager.getKeystorePass(this, path)
        
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 20, 50, 0)
        }
        val etPass = EditText(this).apply { 
            hint = "비밀번호 입력"; inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            setText(savedPass)
        }
        val etAlias = EditText(this).apply { hint = "별칭 (Alias)"; setText("key0") }
        val cbSave = CheckBox(this).apply { text = "비밀번호 기억"; isChecked = savedPass.isNotEmpty() }
        
        layout.addView(etPass); layout.addView(etAlias); layout.addView(cbSave)

        AlertDialog.Builder(this).setTitle("키스토어 인증").setView(layout)
            .setPositiveButton("빌드 시작") { _, _ ->
                val p = etPass.text.toString()
                val a = etAlias.text.toString()
                if (cbSave.isChecked) PrefsManager.saveKeystorePass(this, path, p)
                else PrefsManager.saveKeystorePass(this, path, "")
                
                val config = BuildManager.SigningConfig(File(path), p, a, p)
                executeBuild(project, config)
            }
            .setNegativeButton("취소", null).show()
    }

    private fun showKeystoreDetailsDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 20, 50, 0)
        }
        val etPass = EditText(this).apply { 
            hint = "새 비밀번호 (6자 이상)"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD 
        }
        val etAlias = EditText(this).apply { hint = "별칭 (예: key0)"; setText("key0") }
        val etDName = EditText(this).apply { hint = "조직 정보 (예: CN=AIDE,O=AICode)"; setText("CN=AIDE,O=AICode") }

        layout.addView(etPass); layout.addView(etAlias); layout.addView(etDName)

        val dialog = AlertDialog.Builder(this)
            .setTitle("새 키스토어 생성")
            .setView(layout)
            .setPositiveButton("저장 위치 선택", null)
            .setNegativeButton("취소", null)
            .create()

        dialog.setOnShowListener {
            val btn = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            btn.isEnabled = false

            etPass.addTextChangedListener(object : TextWatcher {
                override fun afterTextChanged(s: Editable?) {
                    btn.isEnabled = (s?.length ?: 0) >= 6
                }
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            })

            btn.setOnClickListener {
                val p = etPass.text.toString()
                val a = etAlias.text.toString()
                val d = etDName.text.toString()
                pendingKeystore = PendingKeystore(p, a, d)
                createKeystoreLauncher.launch("my.keystore")
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun getPathFromUri(uri: Uri): String? {
        if (uri.scheme == "file") return uri.path
        
        return try {
            val f = File(filesDir, "temp_keystore")
            contentResolver.openInputStream(uri)?.use { input ->
                f.outputStream().use { output -> input.copyTo(output) }
            }
            f.absolutePath
        } catch (e: Exception) {
            null
        }
    }

    private fun executeBuild(project: File, signingConfig: BuildManager.SigningConfig?) {
        codeEditor.saveFile()
        btnRun.isEnabled = false
        btnRun.text = "빌드 중..."
        
        if (!logExpanded) toggleLogPanel()

        buildMgr.build(project, signingConfig, object : BuildManager.BuildCallback {
            override fun onProgress(step: String, percent: Int) {
                mainHandler.post { tvStatus.text = "빌드: $step ($percent%)" }
            }

            override fun onComplete(result: BuildManager.BuildResult) {
                mainHandler.post {
                    btnRun.isEnabled = true
                    btnRun.text = "RUN"
                    if (result.success && result.apkFile != null) {
                        tvStatus.text = "빌드 성공!"
                        showInstallDialog(result.apkFile)
                    } else {
                        tvStatus.text = "빌드 실패"
                        Toast.makeText(this@MainActivity, "빌드 실패: ${result.error}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        })
    }

    private fun showInstallDialog(apk: File) {
        AlertDialog.Builder(this).setTitle("빌드 완료")
            .setMessage("APK가 생성되었습니다. 설치하시겠습니까?\n(${apk.name})")
            .setPositiveButton("설치") { _, _ -> buildMgr.installApk(apk) }
            .setNegativeButton("취소", null).show()
    }

    // ─── AI ───────────────────────────────────────

    private fun sendAICommand() {
        val cmd = etAiInput.text.toString().trim()
        if (cmd.isEmpty()) return

        etAiInput.text.clear()
        logger.logUser(cmd)
        btnSendAI.visibility = View.GONE
        btnStopAI.visibility = View.VISIBLE

        aiAgent.execute(cmd)
    }

    private fun toggleLogPanel() {
        logExpanded = !logExpanded
        logPanel.visibility = if (logExpanded) View.VISIBLE else View.GONE
        findViewById<ImageButton>(R.id.btnToggleLog).rotation = if (logExpanded) 0f else 180f
    }

    override fun onDestroy() {
        super.onDestroy()
        // 액티비티 종료 시 바인드만 해제 (서비스는 계속 실행 — 모델 유지)
        if (PrefsManager.getLocalAIMode(this)) {
            localAIMgr.disconnect()
        }
    }

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (drawer.isDrawerOpen(GravityCompat.START)) {
            drawer.closeDrawer(GravityCompat.START)
        } else {
            super.onBackPressed()
        }
    }
}