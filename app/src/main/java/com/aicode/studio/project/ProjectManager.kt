package com.aicode.studio.project

import android.content.Context
import org.json.JSONObject
import java.io.File

class ProjectManager(private val ctx: Context) {

    enum class Language(val label: String) { JAVA("Java"), KOTLIN("Kotlin") }
    enum class Template(val label: String, val assetDir: String) {
        ANDROIDX("AndroidX Project", "AndroidxActivity"),
        EMPTY("Empty Project", "EmptyActivity"),
        COMPOSE("Jetpack Compose", "JetpackCompose")
    }

    data class ProjectConfig(
        val name: String,
        val packageName: String,
        val minSdk: Int = 21,
        val targetSdk: Int = 29, // 34에서 29로 조정 (설치 호환성)
        val language: Language = Language.JAVA,
        val template: Template = Template.ANDROIDX
    )

    val projectsDir: File get() = File(ctx.getExternalFilesDir(null), "Projects").apply { mkdirs() }

    fun listProjects(): List<File> =
        projectsDir.listFiles()?.filter { it.isDirectory }?.sortedByDescending { it.lastModified() } ?: emptyList()

    fun createProject(cfg: ProjectConfig): File {
        if (isProjectExists(cfg.name)) throw Exception("이미 존재하는 프로젝트 이름입니다.")
        
        val root = File(projectsDir, cfg.name).apply { mkdirs() }
        val langDir = if (cfg.language == Language.KOTLIN) "kotlin" else "java"
        val templatePath = "templates/${cfg.template.assetDir}/$langDir"

        copyTemplate(templatePath, root, cfg)

        // Project meta
        File(root, ".aide_project.json").writeText(JSONObject().apply {
            put("name", cfg.name); put("package", cfg.packageName)
            put("language", cfg.language.name); put("minSdk", cfg.minSdk)
            put("targetSdk", cfg.targetSdk); put("template", cfg.template.name)
            put("createdAt", System.currentTimeMillis())
        }.toString(2))

        return root
    }

    fun isProjectExists(name: String): Boolean = File(projectsDir, name).exists()
    
    fun isPackageConflict(packageName: String): Boolean {
        return listProjects().any { 
            getConfig(it)?.packageName == packageName
        }
    }

    private fun copyTemplate(assetPath: String, targetDir: File, cfg: ProjectConfig) {
        val assets = ctx.assets
        val list = assets.list(assetPath) ?: return

        for (item in list) {
            val itemAssetPath = if (assetPath.isEmpty()) item else "$assetPath/$item"
            
            // assets.list()는 폴더인 경우 하위 목록을, 파일인 경우 빈 배열을 반환함
            val subItems = assets.list(itemAssetPath)

            if (subItems != null && subItems.isNotEmpty()) {
                // 폴더임: 재귀 호출
                copyTemplate(itemAssetPath, targetDir, cfg)
            } else {
                // 파일이거나 비어있는 폴더임
                try {
                    val langDirName = if (cfg.language == Language.KOTLIN) "kotlin" else "java"
                    val templatePrefix = "templates/${cfg.template.assetDir}/$langDirName/"
                    
                    // assetPath가 템플릿 경로를 포함하는지 확인하고 상대 경로 추출
                    if (!itemAssetPath.contains(templatePrefix)) continue
                    val targetSubPathRaw = itemAssetPath.substring(itemAssetPath.indexOf(templatePrefix) + templatePrefix.length)
                    var targetSubPath = targetSubPathRaw
                    
                    // 파일명 내 패키지/앱이름 치환
                    if (targetSubPath.contains("${'$'}packagename")) {
                        val packagePath = cfg.packageName.replace(".", "/")
                        targetSubPath = targetSubPath.replace("${'$'}packagename", packagePath)
                    }
                    if (targetSubPath.contains("${'$'}appname")) {
                        targetSubPath = targetSubPath.replace("${'$'}appname", cfg.name)
                    }
                    
                    val targetFile = File(targetDir, targetSubPath)
                    targetFile.parentFile?.mkdirs()
                    
                    // 파일인지 확인 후 복사
                    assets.open(itemAssetPath).use { input ->
                        val textExtensions = listOf("java", "kt", "xml", "gradle", "json", "properties", "pro", "gitignore", "bat", "sh")
                        val textFiles = listOf("gradlew", "local.properties", "settings.gradle", "build.gradle")
                        val isTextFile = targetFile.extension.lowercase() in textExtensions || targetFile.name in textFiles || !targetFile.name.contains(".")
                        
                        if (isTextFile && !targetFile.name.endsWith(".jar")) {
                            var content = input.bufferedReader().use { it.readText() }
                            content = content.replace("${'$'}packagename", cfg.packageName)
                            content = content.replace("${'$'}appname", cfg.name)
                            content = content.replace("${"$"}{minSdkVersion}", cfg.minSdk.toString())
                            content = content.replace("${"$"}{targetSdkVersion}", cfg.targetSdk.toString())
                            targetFile.writeText(content)
                        } else {
                            targetFile.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                    }
                } catch (e: Exception) {
                    // 파일 열기 실패 시 (진짜 비어있는 폴더인 경우) 폴더만 생성
                    val langDirName = if (cfg.language == Language.KOTLIN) "kotlin" else "java"
                    val prefix = "templates/${cfg.template.assetDir}/$langDirName/"
                    if (itemAssetPath.startsWith(prefix)) {
                        var targetSubPath = itemAssetPath.substring(prefix.length)
                        if (targetSubPath.contains("${'$'}packagename")) {
                            targetSubPath = targetSubPath.replace("${'$'}packagename", cfg.packageName.replace(".", "/"))
                        }
                        File(targetDir, targetSubPath).mkdirs()
                    }
                }
            }
        }
    }

    fun deleteProject(dir: File) = dir.deleteRecursively()

    fun getConfig(dir: File): ProjectConfig? {
        val f = File(dir, ".aide_project.json")
        if (!f.exists()) return null
        return try {
            val j = JSONObject(f.readText())
            ProjectConfig(j.getString("name"), j.getString("package"),
                j.optInt("minSdk", 24), j.optInt("targetSdk", 34),
                Language.valueOf(j.optString("language", "JAVA")),
                Template.valueOf(j.optString("template", "ANDROIDX")))
        } catch (_: Exception) { null }
    }

    fun collectSourceFiles(root: File): List<File> =
        root.walkTopDown()
            .filter { it.isFile && !it.path.contains("/build/") && !it.name.startsWith(".") }
            .filter { it.extension.lowercase() in listOf("java", "kt", "xml", "gradle", "json", "properties") }
            .sortedBy { it.absolutePath }.toList()
}