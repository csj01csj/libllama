package com.aicode.studio.build

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.aicode.studio.util.LogManager
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

class BuildManager(private val ctx: Context, private val logger: LogManager) {

    data class BuildResult(
        val success: Boolean,
        val apkFile: File? = null,
        val error: String? = null,
        val durationMs: Long = 0
    )

    data class SigningConfig(
        val keystoreFile: File,
        val storePass: String,
        val alias: String,
        val keyPass: String
    )

    interface BuildCallback {
        fun onProgress(step: String, percent: Int)
        fun onComplete(result: BuildResult)
    }

    private val binDir get() = File(ctx.filesDir, "bin")
    private val aapt2Path get(): String {
        val nativeLib = File(ctx.applicationInfo.nativeLibraryDir, "libaapt2.so")
        if (nativeLib.exists()) {
            // 시스템 추출 경로는 보통 실행 권한이 있으므로 우선 사용
            return nativeLib.absolutePath
        }
        // 차선책으로 내부 bin 폴더 사용
        return File(binDir, "libaapt2.so").absolutePath
    }

    private val javacJar get() = File(binDir, "javac.jar").absolutePath
    private val kotlincJar get() = File(binDir, "kotlinc.jar").absolutePath
    private val d8Jar get() = File(binDir, "d8.jar").absolutePath
    private val androidJar get() = File(binDir, "android.jar").absolutePath
    private val rtJar get() = File(binDir, "rt.jar").absolutePath
    private val lambdaStubsJar get() = File(binDir, "core-lambda-stubs.jar").absolutePath

    fun prepareTools(onReady: () -> Unit) {
        Thread {
            try {
                val dir = binDir
                if (!dir.exists()) dir.mkdirs()
                
                val toolsMap = mapOf(
                    "d8.jar" to "d8.jar",
                    "javac.jar" to "javac.jar",
                    "kotlinc.jar" to "kotlinc.jar",
                    "android.jar" to "android.jar",
                    "apksigner.jar" to "apksigner.jar",
                    "rt.jar" to "rt.jar",
                    "core-lambda-stubs.jar" to "core-lambda-stubs.jar",
                    "libaapt2.so" to "libaapt2.so",
                    "debug.keystore" to "debug.keystore"
                )
                for ((assetName, targetName) in toolsMap) {
                    val outFile = File(dir, targetName)
                    if (!outFile.exists() || targetName == "libaapt2.so" || targetName == "javac.jar" || targetName == "rt.jar" || !targetName.endsWith(".jar")) {
                        try {
                            if (outFile.exists()) outFile.setWritable(true)
                            ctx.assets.open("bin/$assetName").use { input ->
                                FileOutputStream(outFile).use { output ->
                                    input.copyTo(output)
                                }
                            }
                        } catch (e: Exception) {
                            if (assetName == "apksigner.jar") continue
                            else throw e
                        }
                    }
                    if (targetName.endsWith(".so")) {
                        setExecutablePermission(outFile)
                    } else if (targetName.endsWith(".jar")) {
                        outFile.setWritable(false, false)
                    }
                }

                val testResult = execDirect(listOf(aapt2Path, "version"))
                if (testResult.code == 0) {
                    logger.logBuild("aapt2 verified: ${testResult.out.trim()}")
                } else {
                    logger.logError("aapt2 verify failed: ${testResult.err}")
                }
                
                logger.logBuild("Build tools ready")
                onReady()
            } catch (e: Exception) {
                logger.logError("Tool prep failed: ${e.message}")
            }
        }.start()
    }

    private fun setExecutablePermission(file: File) {
        try {
            file.setReadable(true, false)
            file.setExecutable(true, false)
            Runtime.getRuntime().exec(arrayOf("chmod", "755", file.absolutePath)).waitFor()
        } catch (_: Exception) {}
    }

    private data class ExecResult(val code: Int, val out: String, val err: String)

    private fun execDirect(cmdList: List<String>, workingDir: File? = null): ExecResult {
        return try {
            val pb = ProcessBuilder(cmdList)
            pb.directory(workingDir ?: binDir)
            val systemLibPath = "/system/lib64:/vendor/lib64:/system/lib:/vendor/lib"
            val appLibPath = ctx.applicationInfo.nativeLibraryDir
            pb.environment()["LD_LIBRARY_PATH"] = "$appLibPath:$binDir:$systemLibPath"
            
            val p = pb.start()
            
            val outLines = mutableListOf<String>()
            val errLines = mutableListOf<String>()
            
            val outThread = Thread {
                p.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        val l = line.lowercase()
                        if (!l.contains("metadata") && !l.contains("unexpected error") && !l.contains("invalid locals")) {
                            outLines.add(line)
                            logger.logBuild(line)
                        }
                    }
                }
            }
            val errThread = Thread {
                p.errorStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        val l = line.lowercase()
                        if (!l.contains("metadata") && !l.contains("unexpected error") && !l.contains("invalid locals")) {
                            errLines.add(line)
                        }
                    }
                }
            }
            
            outThread.start()
            errThread.start()
            
            val code = p.waitFor()
            outThread.join() // 무제한 대기 (프로세스가 종료되면 스트림도 닫힘)
            errThread.join()
            
            val out = outLines.joinToString("\n")
            val err = errLines.joinToString("\n")
            
            if (err.isNotBlank()) {
                if (code == 0) logger.logBuild(err.trim())
                else logger.logError(err.trim())
            }
            
            ExecResult(code, out, err)
        } catch (e: Exception) {
            ExecResult(-1, "", e.message ?: "Execution failed")
        }
    }

    private fun exec(cmd: String): ExecResult {
        val args = cmd.split(" ").filter { it.isNotBlank() }
        return execDirect(args)
    }

    fun build(projectRoot: File, signingConfig: SigningConfig?, callback: BuildCallback) {
        Thread {
            val t0 = System.currentTimeMillis()
            try {
                val res = File(projectRoot, "app/src/main/res")
                val src = File(projectRoot, "app/src/main/java")
                val manifest = File(projectRoot, "app/src/main/AndroidManifest.xml")
                val buildDir = File(projectRoot, "build").apply { deleteRecursively(); mkdirs() }
                val genDir = File(buildDir, "gen").apply { mkdirs() }
                val classDir = File(buildDir, "classes").apply { mkdirs() }

                if (!manifest.exists()) throw B("AndroidManifest.xml 없음")
                if (!src.exists()) throw B("소스 디렉토리 없음")

                // Step 1: aapt2 compile
                callback.onProgress("리소스 컴파일...", 10)
                logger.logBuild("=== Step 1: AAPT2 Compile ===")
                val resZip = File(buildDir, "compiled_res.zip")
                val compiledResFiles = mutableListOf<File>()

                // 1.1 프로젝트 리소스 컴파일
                if (res.exists() && (res.listFiles()?.isNotEmpty() == true)) {
                    val r = execDirect(listOf(aapt2Path, "compile", "--dir", res.absolutePath, "-o", resZip.absolutePath))
                    if (r.code != 0) throw B("aapt2 compile 실패:\n${r.err}")
                    compiledResFiles.add(resZip)
                    logger.logBuild("프로젝트 리소스 컴파일 완료")
                }

                // 1.2 라이브러리 처리 (필터링된 리스트 사용)
                val libExtractDir = File(buildDir, "libs_extracted").apply { mkdirs() }
                val projectLibs = mutableListOf<File>()
                val aarPackages = mutableListOf<String>()
                val libraryManifests = mutableListOf<File>()
                val mergedAssetsDir = File(buildDir, "merged_assets").apply { mkdirs() }
                
                // 검색할 파일 리스트 준비
                val targetLibFiles = mutableListOf<File>()
                
                // 1. resolved_deps.txt 읽기 (SYNC 결과)
                val resolvedDepsFile = File(projectRoot, "app/resolved_deps.txt")
                if (resolvedDepsFile.exists()) {
                    resolvedDepsFile.readLines().forEach { path ->
                        val f = File(path)
                        if (f.exists()) targetLibFiles.add(f)
                    }
                }
                
                // 2. 프로젝트 내부 libs 폴더 추가 (수동 추가 파일)
                val internalLibsDir = File(projectRoot, "app/libs")
                if (internalLibsDir.exists()) {
                    internalLibsDir.walkTopDown().forEach {
                        if (it.isFile && (it.extension == "jar" || it.extension == "aar")) {
                            targetLibFiles.add(it)
                        }
                    }
                }

                // 중복 제거 및 처리
                targetLibFiles.distinctBy { it.absolutePath }.forEach { it ->
                    if (it.extension.equals("jar", true)) {
                        if (!it.name.contains("gradle-wrapper")) {
                            projectLibs.add(it)
                        }
                    } else if (it.extension.equals("aar", true)) {
                        logger.logBuild("AAR 처리: ${it.name}")
                        val targetDir = File(libExtractDir, it.nameWithoutExtension).apply { mkdirs() }
                        extractZip(it, targetDir)
                        
                        // 매니페스트 수집
                        val am = File(targetDir, "AndroidManifest.xml")
                        if (am.exists()) {
                            libraryManifests.add(am)
                            val pkgMatch = Regex("package=\"([^\"]+)\"").find(am.readText())
                            pkgMatch?.let { m -> aarPackages.add(m.groupValues[1]) }
                        }

                        // 에셋 수집
                        val la = File(targetDir, "assets")
                        if (la.exists()) {
                            la.copyRecursively(mergedAssetsDir, overwrite = true)
                        }

                        val cj = File(targetDir, "classes.jar")
                        if (cj.exists()) {
                            val renamedJar = File(targetDir, "${it.nameWithoutExtension}-classes.jar")
                            cj.renameTo(renamedJar)
                            projectLibs.add(renamedJar)
                        }
                        
                        val lr = File(targetDir, "res")
                        if (lr.exists()) {
                            val libResOut = File(buildDir, "res_lib_${it.nameWithoutExtension}").apply { mkdirs() }
                            val r = execDirect(listOf(aapt2Path, "compile", "--dir", lr.absolutePath, "-o", libResOut.absolutePath))
                            if (r.code == 0) {
                                libResOut.listFiles()?.forEach { flat ->
                                    if (flat.isFile && (flat.extension == "flat" || flat.extension == "zip")) {
                                        compiledResFiles.add(flat)
                                    }
                                }
                            }
                        }
                    }
                }

                // 매니페스트 병합 로직 (간이 구현)
                val finalManifest = File(buildDir, "merged_manifest.xml")
                mergeManifests(manifest, libraryManifests, finalManifest)

                // Step 2: aapt2 link
                callback.onProgress("리소스 링크...", 25)
                logger.logBuild("=== Step 2: AAPT2 Link ===")
                val linked = File(buildDir, "linked.apk")
                val linkArgs = mutableListOf(
                    aapt2Path, "link", "-o", linked.absolutePath,
                    "-I", androidJar,
                    "--manifest", finalManifest.absolutePath, // 병합된 매니페스트 사용
                    "--java", genDir.absolutePath,
                    "--auto-add-overlay"
                )
                // 프로젝트 에셋 + 병합된 라이브러리 에셋
                val projectAssets = File(projectRoot, "app/src/main/assets")
                if (projectAssets.exists()) projectAssets.copyRecursively(mergedAssetsDir, overwrite = true)
                if (mergedAssetsDir.exists() && mergedAssetsDir.list()?.isNotEmpty() == true) {
                    linkArgs.add("-A")
                    linkArgs.add(mergedAssetsDir.absolutePath)
                }

                if (aarPackages.isNotEmpty()) {
                    linkArgs.add("--extra-packages")
                    linkArgs.add(aarPackages.distinct().joinToString(":"))
                }
                compiledResFiles.forEach { linkArgs.add(it.absolutePath) }
                
                val lr = execDirect(linkArgs)
                if (lr.code != 0) throw B("aapt2 link 실패:\n${lr.err}")
                logger.logBuild("R.java 생성 완료")

                // Step 3: Compile (Java & Kotlin)
                callback.onProgress("컴파일 중...", 45)
                logger.logBuild("=== Step 3: Compile (Java/Kotlin) ===")
                val javaSources = mutableListOf<File>()
                val kotlinSources = mutableListOf<File>()
                
                logger.logBuild("소스 검색 중...")
                projectRoot.walkTopDown().forEach {
                    if (it.isFile && !it.path.contains("/build/")) {
                        val relPath = it.relativeTo(projectRoot).path
                        if (it.extension.equals("java", true)) {
                            javaSources.add(it)
                            logger.logBuild("찾음(Java): $relPath")
                        } else if (it.extension.equals("kt", true)) {
                            kotlinSources.add(it)
                            logger.logBuild("찾음(Kotlin): $relPath")
                        }
                    }
                }
                
                // 생성된 R.java 추가
                if (genDir.exists()) {
                    genDir.walkTopDown().forEach {
                        if (it.isFile && it.extension.equals("java", true)) javaSources.add(it)
                    }
                }

                if (javaSources.isEmpty() && kotlinSources.isEmpty()) throw B("소스 파일 없음")
                
                val allSources = javaSources + kotlinSources
                val srcList = File(buildDir, "sources.txt")
                srcList.writeText(allSources.joinToString("\n") { it.absolutePath })

                val tmpDir = File(buildDir, "tmp").apply { mkdirs() }
                
                val libCP = projectLibs.joinToString(":") { it.absolutePath }
                val fullCP = if (libCP.isNotEmpty()) "$androidJar:$rtJar:$lambdaStubsJar:$libCP" else "$androidJar:$rtJar:$lambdaStubsJar"

                // Kotlin 파일이 있으면 kotlinc 우선 실행
                if (kotlinSources.isNotEmpty()) {
                    logger.logBuild("Kotlin 소스 감지됨 (${kotlinSources.size}개). 코틀린 컴파일 시작...")
                    val kotlincArgs = mutableListOf(
                        "dalvikvm", "-Xmx1024m", "-Djava.io.tmpdir=${tmpDir.absolutePath}",
                        "-cp", kotlincJar,
                        "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
                        "-no-jdk",
                        "-cp", fullCP, // 라이브러리 포함
                        "-d", classDir.absolutePath
                    )
                    allSources.forEach { kotlincArgs.add(it.absolutePath) }
                    
                    val kr = execDirect(kotlincArgs)
                    if (kr.code != 0) {
                        logger.logError("Kotlin 컴파일 실패 (Code ${kr.code})")
                        throw B("Kotlin 컴파일 실패:\n${kr.err}")
                    }
                }

                // Java 파일 컴파일
                if (javaSources.isNotEmpty()) {
                    logger.logBuild("Java 컴파일 시작 (${javaSources.size}개)...")
                    val javacArgs = mutableListOf(
                        "dalvikvm", "-Xmx1024m", "-Djava.io.tmpdir=${tmpDir.absolutePath}",
                        "-cp", "$javacJar:$rtJar:$lambdaStubsJar",
                        "com.sun.tools.javac.Main",
                        "-d", classDir.absolutePath,
                        "-bootclasspath", "$rtJar:$lambdaStubsJar",
                        "-classpath", "$fullCP:${classDir.absolutePath}", // 라이브러리 및 Kotlin 결과물 포함
                        "-source", "1.8",
                        "-target", "1.8",
                        "-proc:none"
                    )
                    javaSources.forEach { javacArgs.add(it.absolutePath) }
                    
                    val jr = execDirect(javacArgs)
                    if (jr.code != 0) throw B("Java 컴파일 실패")
                }
                
                Thread.sleep(300)
                val compiledClasses = mutableListOf<File>()
                classDir.walkTopDown().forEach { 
                    if(it.isFile && it.extension == "class") compiledClasses.add(it)
                }
                logger.logBuild("컴파일 완료: ${compiledClasses.size}개 .class 파일 생성됨")
                
                if (compiledClasses.isEmpty()) throw B("생성된 .class 파일이 없습니다.")

                // Step 4: D8
                callback.onProgress("DEX 변환...", 65)
                logger.logBuild("=== Step 4: D8 DEX ===")
                logger.logBuild("DEX 입력: ${compiledClasses.size}개 클래스, ${projectLibs.size}개 라이브러리")

                val dexOut = File(buildDir, "dex").apply { mkdirs() }
                val d8Args = mutableListOf(
                    "dalvikvm", "-Xmx1024m", "-Djava.io.tmpdir=${tmpDir.absolutePath}",
                    "-cp", d8Jar,
                    "com.android.tools.r8.D8",
                    "--output", dexOut.absolutePath,
                    "--min-api", "24",
                    "--lib", androidJar,
                    "--lib", rtJar,
                    "--lib", lambdaStubsJar
                )
                // 프로젝트 클래스들 추가
                compiledClasses.forEach { d8Args.add(it.absolutePath) }
                // 모든 라이브러리 JAR 추가 (이게 빠지면 ClassNotFound 발생)
                projectLibs.forEach { d8Args.add(it.absolutePath) }

                
                val dr = execDirect(d8Args)
                if (dr.code != 0) throw B("D8 DEX 변환 실패:\n${dr.err}")
                logger.logBuild("DEX 변환 완료")

                // Step 5: APK Package
                callback.onProgress("APK 패키징...", 85)
                logger.logBuild("=== Step 5: APK Package ===")
                val unsigned = File(buildDir, "unsigned.apk")
                mergeApk(linked, dexOut, unsigned)
                logger.logBuild("APK 패키징 완료")


                // Step 6: Sign APK
                callback.onProgress("APK 서명...", 90)
                logger.logBuild("=== Step 6: Sign APK ===")
                val signed = File(buildDir, "app-release.apk")
                if (signingConfig != null) {
                    customSign(unsigned, signed, signingConfig)
                } else {
                    debugSign(unsigned, signed)
                }
                logger.logBuild("서명 완료")

                val dur = System.currentTimeMillis() - t0
                logger.logBuild("=== BUILD SUCCESSFUL (${dur}ms) ===")
                
                // APK 백업 로직 추가
                try {
                    val backupDir = File("/storage/emulated/0/aistudio")
                    if (!backupDir.exists()) backupDir.mkdirs()
                    val backupFile = File(backupDir, "${projectRoot.name}.apk")
                    signed.copyTo(backupFile, overwrite = true)
                    logger.logSystem("백업 완료: ${backupFile.absolutePath}")
                } catch (e: Exception) {
                    logger.logError("백업 실패: ${e.message}")
                }

                callback.onProgress("완료!", 100)
                callback.onComplete(BuildResult(true, signed, durationMs = dur))

            } catch (e: B) {
                logger.logError("BUILD FAILED: ${e.message}")
                callback.onComplete(BuildResult(false, error = e.message, durationMs = System.currentTimeMillis() - t0))
            } catch (e: Exception) {
                logger.logError("BUILD EXCEPTION: ${e.message}")
                callback.onComplete(BuildResult(false, error = e.message, durationMs = System.currentTimeMillis() - t0))
            }
        }.start()
    }

    fun installApk(apkFile: File) {
        try {
            logger.logSystem("설치 시도 중: ${apkFile.name}")
            val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.provider", apkFile)
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "application/vnd.android.package-archive")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            ctx.startActivity(intent)
            logger.logSystem("시스템 설치 화면으로 이동했습니다.")
        } catch (e: Exception) {
            logger.logError("설치 실패: ${e.message}")
        }
    }

    private class B(msg: String) : Exception(msg)

    private fun mergeManifests(main: File, libs: List<File>, output: File) {
        var mainXml = main.readText()
        
        // Ensure standard namespaces are defined at the root to avoid "unbound prefix" errors
        val namespaces = mapOf(
            "xmlns:android" to "http://schemas.android.com/apk/res/android",
            "xmlns:tools" to "http://schemas.android.com/tools",
            "xmlns:app" to "http://schemas.android.com/apk/res-auto"
        )
        
        var modifiedMain = mainXml
        namespaces.forEach { (prefix, uri) ->
            if (!modifiedMain.contains(prefix)) {
                modifiedMain = modifiedMain.replaceFirst("<manifest", "<manifest $prefix=\"$uri\"")
            }
        }
        
        val appTagEnd = modifiedMain.indexOf("</application>")
        if (appTagEnd == -1) {
            output.writeText(modifiedMain)
            return
        }

        val sb = StringBuilder(modifiedMain.substring(0, appTagEnd))
        libs.forEach { lf ->
            try {
                val lxml = lf.readText()
                // Extract contents within <application> (activities, services, etc.)
                val start = lxml.indexOf("<application")
                if (start != -1) {
                    val contentStart = lxml.indexOf(">", start) + 1
                    val contentEnd = lxml.indexOf("</application>")
                    if (contentEnd > contentStart) {
                        val components = lxml.substring(contentStart, contentEnd)
                        sb.append("\n        <!-- From Library: ${lf.parentFile.name} -->\n")
                        sb.append(components)
                    }
                }
            } catch (_: Exception) {}
        }
        sb.append("\n    </application>\n</manifest>")
        output.writeText(sb.toString())
    }

    private fun extractZip(zipFile: File, targetDir: File) {
        ZipFile(zipFile).use { zip ->
            zip.entries().asSequence().forEach { entry ->
                val f = File(targetDir, entry.name)
                if (entry.isDirectory) {
                    f.mkdirs()
                } else {
                    f.parentFile?.mkdirs()
                    zip.getInputStream(entry).use { it.copyTo(f.outputStream()) }
                }
            }
        }
    }

    private fun mergeApk(linkedApk: File, dexDir: File, output: File) {
        ZipOutputStream(FileOutputStream(output)).use { zos ->
            ZipFile(linkedApk).use { zf ->
                for (entry in zf.entries()) {
                    val newEntry = ZipEntry(entry.name)
                    newEntry.method = entry.method
                    if (newEntry.method == ZipEntry.STORED || entry.name == "resources.arsc") {
                        newEntry.method = ZipEntry.STORED
                        newEntry.size = entry.size
                        newEntry.compressedSize = entry.size
                        newEntry.crc = entry.crc
                    }
                    zos.putNextEntry(newEntry)
                    zf.getInputStream(entry).use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
            // 모든 DEX 파일 추가 (MultiDex 지원)
            dexDir.listFiles()?.filter { it.extension == "dex" }?.sortedBy { it.name }?.forEach { df ->
                val dexEntry = ZipEntry(df.name)
                zos.putNextEntry(dexEntry)
                FileInputStream(df).use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
    }

    private fun debugSign(input: File, output: File) {
        val ksFile = File(binDir, "debug.keystore")
        if (!ksFile.exists()) throw Exception("debug.keystore 파일이 누락되었습니다.")
        val config = SigningConfig(ksFile, "android", "androiddebugkey", "android")
        customSign(input, output, config)
    }

    private fun customSign(input: File, output: File, config: SigningConfig) {
        if (!config.keystoreFile.exists()) throw Exception("키스토어 파일을 찾을 수 없습니다: ${config.keystoreFile.name}")
        
        input.copyTo(output, overwrite = true)
        val apksigner = File(binDir, "apksigner.jar")
        if (apksigner.exists()) {
            val tmpDir = File(ctx.cacheDir, "apksigner_tmp").apply { mkdirs() }
            val cmd = "dalvikvm -Xmx1024m -Djava.io.tmpdir=${tmpDir.absolutePath} -cp ${apksigner.absolutePath} com.android.apksigner.ApkSignerTool sign " +
                    "--ks ${config.keystoreFile.absolutePath} " +
                    "--ks-pass pass:${config.storePass} " +
                    "--key-pass pass:${config.keyPass} " +
                    "--ks-key-alias ${config.alias} " +
                    "--ks-type PKCS12 " +
                    "--v1-signing-enabled true " + // V1 서명 활성화
                    "--v2-signing-enabled true " + // V2 서명 활성화 (안드로이드 11+ 필수)
                    output.absolutePath
            
            val er = exec(cmd)
            if (er.code != 0) throw Exception("APK 서명 실패:\n${er.err}")
        } else {
            throw Exception("apksigner.jar를 찾을 수 없습니다.")
        }
    }

    // 진짜 키스토어 생성 (시스템 keytool 호출)
    fun createKeystore(file: File, pass: String, alias: String, dname: String, callback: (Boolean, String) -> Unit) {
        Thread {
            try {
                // 안드로이드 기기 내 keytool 경로는 보통 /system/bin/keytool 임
                val cmd = mutableListOf(
                    "keytool", "-genkey", "-v",
                    "-keystore", file.absolutePath,
                    "-storepass", pass,
                    "-alias", alias,
                    "-keypass", pass,
                    "-keyalg", "RSA",
                    "-keysize", "2048",
                    "-validity", "10000",
                    "-dname", dname,
                    "-storetype", "PKCS12"
                )
                
                // 쉘을 통해 실행하여 시스템 PATH를 활용
                val pb = ProcessBuilder(cmd)
                val env = pb.environment()
                env["PATH"] = "/system/bin:/system/xbin:/vendor/bin:" + (env["PATH"] ?: "")
                
                val p = pb.start()
                val err = p.errorStream.bufferedReader().readText()
                val code = p.waitFor()
                
                if (code == 0) {
                    callback(true, "성공")
                } else {
                    callback(false, "키스토어 생성 실패 (Code $code):\n$err")
                }
            } catch (e: Exception) {
                callback(false, "에러 발생: ${e.message}")
            }
        }.start()
    }
}