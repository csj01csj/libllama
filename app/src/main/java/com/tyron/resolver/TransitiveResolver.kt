package com.tyron.resolver

import okhttp3.OkHttpClient
import okhttp3.Request
import org.w3c.dom.Element
import org.w3c.dom.Node
import java.io.ByteArrayInputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.xml.parsers.DocumentBuilderFactory

/**
 * Maven 전이 의존성 리졸버 (병렬 분석 및 안드로이드 최적화 필터링)
 */
class TransitiveResolver(private val client: OkHttpClient) {

    data class Dep(val g: String, val a: String, val v: String, val scope: String? = null, val optional: Boolean = false) {
        override fun toString() = "$g:$a:$v"
        fun toPath(ext: String) = "${g.replace(".", "/")}/$a/$v/$a-$v.$ext"
    }

    private val repos = listOf(
        "https://maven.google.com/",
        "https://repo1.maven.org/maven2/",
        "https://jcenter.bintray.com/"
    )

    // 안드로이드 빌드에 불필요한 전이 의존성 그룹 블랙리스트
    private val blacklistedGroups = listOf(
        "org.apache.maven",
        "org.codehaus.plexus",
        "org.sonatype",
        "org.eclipse.aether",
        "org.apache.ant",
        "org.apache.uima",
        "org.apache.solr",
        "org.apache.lucene",
        "org.apache.hadoop",
        "org.jboss.scm",
        "org.apache.maven.scm"
    )

    private val resolved = ConcurrentHashMap<String, Dep>()
    private val globalProperties = ConcurrentHashMap<String, String>()
    private val pomCache = ConcurrentHashMap<String, String>()
    private val visiting = ConcurrentHashMap.newKeySet<String>()
    
    var onProgress: ((String) -> Unit)? = null

    fun resolve(roots: List<Dep>): List<Dep> {
        resolved.clear()
        globalProperties.clear()
        pomCache.clear()
        visiting.clear()

        // 병렬 처리를 위한 스레드 풀 (최대 8개 스레드)
        val executor = Executors.newFixedThreadPool(8)
        
        roots.forEach { root ->
            executor.execute {
                onProgress?.invoke("분석 중: ${root.a}")
                recursiveResolve(root, executor)
            }
        }
        
        executor.shutdown()
        executor.awaitTermination(10, TimeUnit.MINUTES)
        
        // Kotlin 1.8.0+ 대응
        val stdlib = resolved["org.jetbrains.kotlin:kotlin-stdlib"]
        if (stdlib != null && compareVersions(stdlib.v, "1.8.0") >= 0) {
            resolved.remove("org.jetbrains.kotlin:kotlin-stdlib-jdk7")
            resolved.remove("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
        }
        
        return resolved.values.toList()
    }

    private fun recursiveResolve(dep: Dep, executor: java.util.concurrent.ExecutorService) {
        // 1. 기본 필터링
        if (dep.scope in listOf("test", "provided", "system")) return
        if (dep.optional) return
        
        // 안드로이드 불필요 그룹 필터링 (전이 의존성만)
        if (blacklistedGroups.any { dep.g.startsWith(it) }) return

        val cleanV = dep.v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")[0].trim()
        if (cleanV.isEmpty() || cleanV.contains("$")) return
        
        val cleanDep = dep.copy(v = cleanV)
        val key = "${cleanDep.g}:${cleanDep.a}"
        
        // 이미 방문 중이거나 더 높은 버전이 있으면 스킵
        if (!visiting.add(key + ":" + cleanV)) return
        
        val existing = resolved[key]
        if (existing != null) {
            if (compareVersions(cleanDep.v, existing.v) > 0) {
                resolved[key] = cleanDep
            } else {
                visiting.remove(key + ":" + cleanV)
                return 
            }
        } else {
            resolved[key] = cleanDep
        }

        // POM 가져오기 및 분석
        val pomContent = fetchPom(cleanDep)
        if (pomContent != null) {
            val pomData = parsePomRecursively(pomContent, 0)
            pomData.dependencies.forEach { child ->
                executor.execute { recursiveResolve(child, executor) }
            }
        }
        
        visiting.remove(key + ":" + cleanV)
    }

    private fun parsePomRecursively(xml: String, depth: Int): PomData {
        val data = PomData()
        if (depth > 10) return data
        
        try {
            val db = DocumentBuilderFactory.newInstance().newDocumentBuilder()
            val doc = db.parse(ByteArrayInputStream(xml.toByteArray()))
            
            val ownG = doc.getElementsByTagName("groupId").item(0)?.textContent?.trim()
            val ownA = doc.getElementsByTagName("artifactId").item(0)?.textContent?.trim()
            val ownV = doc.getElementsByTagName("version").item(0)?.textContent?.trim()

            // 1. Parent POM 처리
            val parentNode = doc.getElementsByTagName("parent").item(0) as? Element
            if (parentNode != null) {
                val pg = parentNode.getElementsByTagName("groupId").item(0)?.textContent?.trim()
                val pa = parentNode.getElementsByTagName("artifactId").item(0)?.textContent?.trim()
                val pv = parentNode.getElementsByTagName("version").item(0)?.textContent?.trim()
                if (pg != null && pa != null && pv != null) {
                    val pDep = Dep(pg, pa, pv)
                    val parentPom = fetchPom(pDep)
                    if (parentPom != null) {
                        val parentData = parsePomRecursively(parentPom, depth + 1)
                        data.properties.putAll(parentData.properties)
                        data.dependencies.addAll(parentData.dependencies)
                        
                        data.properties["project.parent.groupId"] = pg
                        data.properties["project.parent.version"] = pv
                        data.properties["parent.groupId"] = pg
                        data.properties["parent.version"] = pv
                    }
                }
            }

            // 2. 내장 속성
            val finalG = ownG ?: data.properties["project.parent.groupId"] ?: ""
            val finalV = ownV ?: data.properties["project.parent.version"] ?: ""
            if (finalG.isNotEmpty()) {
                data.properties["project.groupId"] = finalG
                data.properties["pom.groupId"] = finalG
                data.properties["groupId"] = finalG
            }
            if (ownA != null) {
                data.properties["project.artifactId"] = ownA
                data.properties["pom.artifactId"] = ownA
                data.properties["artifactId"] = ownA
            }
            if (finalV.isNotEmpty()) {
                data.properties["project.version"] = finalV
                data.properties["pom.version"] = finalV
                data.properties["version"] = finalV
            }

            // 3. Properties 추출
            val propsNode = doc.getElementsByTagName("properties").item(0) as? Element
            if (propsNode != null) {
                val children = propsNode.childNodes
                for (i in 0 until children.length) {
                    val node = children.item(i)
                    if (node.nodeType == Node.ELEMENT_NODE) {
                        val key = node.nodeName
                        val value = node.textContent.trim()
                        data.properties[key] = value
                        globalProperties[key] = value
                    }
                }
            }

            // 4. Dependencies 추출
            val nodes = doc.getElementsByTagName("dependency")
            for (i in 0 until nodes.length) {
                val node = nodes.item(i) as Element
                val g = resolveValue(node.getElementsByTagName("groupId").item(0)?.textContent, data.properties)
                val a = resolveValue(node.getElementsByTagName("artifactId").item(0)?.textContent, data.properties)
                val v = resolveValue(node.getElementsByTagName("version").item(0)?.textContent, data.properties)
                val s = node.getElementsByTagName("scope").item(0)?.textContent?.trim()
                val opt = node.getElementsByTagName("optional").item(0)?.textContent?.trim() == "true"
                
                if (g != null && a != null && v != null) {
                    data.dependencies.add(Dep(g, a, v, s, opt))
                }
            }
        } catch (e: Exception) {}
        return data
    }

    private data class PomData(
        val properties: MutableMap<String, String> = mutableMapOf(),
        val dependencies: MutableList<Dep> = mutableListOf()
    )

    private fun resolveValue(v: String?, props: Map<String, String>): String? {
        if (v == null) return null
        var result = v.trim()
        
        for (i in 0 until 3) {
            val start = result.indexOf("\${")
            val end = result.indexOf("}", start)
            if (start != -1 && end != -1) {
                val key = result.substring(start + 2, end)
                val replacement = props[key] ?: globalProperties[key]
                if (replacement != null) {
                    result = result.replace("\${$key}", replacement)
                } else break
            } else break
        }
        return result
    }

    private fun compareVersions(v1: String, v2: String): Int {
        val parts1 = v1.split(".", "-").mapNotNull { it.toIntOrNull() }
        val parts2 = v2.split(".", "-").mapNotNull { it.toIntOrNull() }
        val max = maxOf(parts1.size, parts2.size)
        for (i in 0 until max) {
            val p1 = parts1.getOrElse(i) { 0 }
            val p2 = parts2.getOrElse(i) { 0 }
            if (p1 != p2) return p1.compareTo(p2)
        }
        return v1.compareTo(v2)
    }

    private fun fetchPom(dep: Dep): String? {
        val cacheKey = dep.toString()
        pomCache[cacheKey]?.let { return it }

        for (base in repos) {
            try {
                val url = base + dep.toPath("pom")
                val req = Request.Builder().url(url).build()
                client.newCall(req).execute().use { resp ->
                    if (resp.isSuccessful) {
                        val body = resp.body?.string()
                        if (body != null) {
                            pomCache[cacheKey] = body
                            return body
                        }
                    }
                }
            } catch (e: Exception) {}
        }
        return null
    }
}
