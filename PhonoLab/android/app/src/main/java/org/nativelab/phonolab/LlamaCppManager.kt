package org.nativelab.phonolab

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Centralized manager for llama-server binary.
 * Downloads from ggml-org/llama.cpp GitHub releases.
 *
 * Binary is stored in user-selected folder, then COPIED to
 * nativeLibraryDir for execution (Android blocks exec from filesDir).
 */
class LlamaCppManager(
    private val context: Context,
    private val store: PhonoLabStore,
    private val storageManager: StorageManager? = null,
) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("llama_cpp_manager", Context.MODE_PRIVATE)

    data class RuntimeStatus(
        val ready: Boolean,
        val serverPath: String,
        val libDir: String,
        val message: String,
        val version: String = "",
    )

    // ── Architecture detection ──────────────────────────────────────

    data class ArchInfo(
        val primary: String,
        val allAbis: List<String>,
        val isArm64: Boolean,
        val isEmulator: Boolean,
    )

    fun detectArch(): ArchInfo {
        val abis = Build.SUPPORTED_ABIS.toList()
        val primary = abis.firstOrNull() ?: "unknown"
        val isArm64 = abis.any { it == "arm64-v8a" }
        val isEmulator = Build.FINGERPRINT.contains("generic") ||
                Build.FINGERPRINT.contains("emulator") ||
                Build.MODEL.contains("sdk_gphone") ||
                Build.MODEL.contains("Emulator") ||
                Build.HARDWARE.contains("goldfish") ||
                Build.HARDWARE.contains("ranchu") ||
                Build.PRODUCT.contains("sdk") ||
                Build.PRODUCT.contains("emulator")
        return ArchInfo(primary, abis, isArm64, isEmulator)
    }

    fun archDownloadSuffix(): String? {
        val arch = detectArch()
        if (arch.isArm64) return "android-arm64"
        return null
    }

    // ── Status ──────────────────────────────────────────────────────

    fun status(): RuntimeStatus {
        val server = findServer() ?: return RuntimeStatus(false, "", "", "llama-server not installed")
        val libDir = store.runtimeDir.absolutePath
        val version = getBinaryVersion(server, libDir)
        return RuntimeStatus(true, server.absolutePath, libDir, "llama-server ready", version)
    }

    // ── Find / Deploy ───────────────────────────────────────────────

    /**
     * Find llama-server binary.
     * ALWAYS returns the nativeLibraryDir copy (the only place exec() works).
     * If not deployed yet, copies from store.runtimeDir.
     */
    fun findServer(): File? {
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val nativeServer = File(nativeDir, "llama-server")

        // Already deployed?
        if (nativeServer.exists() && nativeServer.length() > 1000) {
            return nativeServer
        }

        // Source exists in user folder — deploy it
        val source = File(store.runtimeDir, "llama-server")
        if (source.exists() && source.length() > 1000) {
            return deployBinary(source)
        }

        // Legacy location
        val legacy = File(store.runtimeBinDir, "llama-server")
        if (legacy.exists() && legacy.length() > 1000 && legacy.absolutePath != source.absolutePath) {
            return deployBinary(legacy)
        }

        return null
    }

    /**
     * Copy binary to nativeLibraryDir. This is the ONLY location
     * where Android allows exec() on most devices.
     */
    private fun deployBinary(source: File): File? {
        return try {
            val nativeDir = File(context.applicationInfo.nativeLibraryDir)
            val target = File(nativeDir, "llama-server")
            nativeDir.mkdirs()

            // Only copy if different
            if (!target.exists() || target.length() != source.length()) {
                source.copyTo(target, overwrite = true)
            }
            target.setExecutable(true, false)

            // Also copy .so files to nativeLibDir
            val sourceDir = source.parentFile
            sourceDir?.listFiles()?.filter { it.name.endsWith(".so") }?.forEach { so ->
                val soTarget = File(nativeDir, so.name)
                if (!soTarget.exists() || soTarget.length() != so.length()) {
                    so.copyTo(soTarget, overwrite = true)
                }
                soTarget.setExecutable(true, false)
            }

            if (target.exists() && target.length() > 1000) target else null
        } catch (_: Exception) {
            null
        }
    }

    // ── Verify ──────────────────────────────────────────────────────

    /** Verify binary is a valid ELF file. */
    fun verifyBinary(server: File, libDir: String): Boolean {
        if (!server.exists() || server.length() < 5000) return false
        return try {
            val magic = server.inputStream().use { it.readNBytes(4) }
            magic.size >= 4 && magic[0] == 0x7f.toByte() && magic[1] == 'E'.code.toByte() &&
                magic[2] == 'L'.code.toByte() && magic[3] == 'F'.code.toByte()
        } catch (_: Exception) { false }
    }

    /** Run a binary with LD_LIBRARY_PATH. */
    private fun runBinary(binary: File, libDir: String, vararg args: String): Pair<Int, String> {
        val env = HashMap(System.getenv())
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val ldPaths = mutableListOf(libDir)
        if (nativeDir != libDir) ldPaths.add(nativeDir)
        val existing = env["LD_LIBRARY_PATH"] ?: ""
        if (existing.isNotEmpty()) ldPaths.add(existing)
        env["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")

        val command = listOf(binary.absolutePath) + args.toList()
        val proc = ProcessBuilder(command)
            .redirectErrorStream(true)
            .apply { environment().putAll(env) }
            .start()
        val output = proc.inputStream.bufferedReader().readText().trim()
        val exitCode = proc.waitFor()
        return Pair(exitCode, output)
    }

    private fun getBinaryVersion(server: File, libDir: String): String {
        return try {
            val (_, output) = runBinary(server, libDir, "--version")
            if (output.isNotBlank()) output.take(200) else "installed"
        } catch (_: Exception) { "installed" }
    }

    // ── Download & Install ──────────────────────────────────────────

    fun downloadAndInstall(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? {
        val dir = store.runtimeDir
        dir.mkdirs()

        // Check architecture
        val arch = detectArch()
        val archSuffix = archDownloadSuffix()
        if (archSuffix == null) {
            error(
                "This device runs ${arch.primary} (${arch.allAbis.joinToString()}). " +
                "llama.cpp only provides arm64-v8a Android binaries. " +
                if (arch.isEmulator) {
                    "Your emulator is x86_64. Use an ARM64 emulator image or a physical arm64 device."
                } else {
                    "This device does not support arm64."
                }
            )
        }

        // Get latest release tag
        val tag = latestReleaseTag()
        if (tag.isEmpty()) error("Could not determine latest llama.cpp release version")

        val url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-$archSuffix.tar.gz"
        val archiveFile = File(store.downloadsDir, "llama-$tag-android-arm64.tar.gz")

        try {
            // Download
            try {
                downloadFile(url, archiveFile, "llama-server", progress)
            } catch (e: Exception) {
                error("Download failed: ${e.message}\nURL: $url")
            }

            // Extract
            try {
                extractTarGz(archiveFile, dir)
            } catch (e: Exception) {
                error("Extraction failed: ${e.message}")
            }

            // Move from extracted subdir to dir
            val extractedDir = File(dir, "llama-$tag")
            val serverInExtracted = File(extractedDir, "llama-server")
            if (!serverInExtracted.exists() || serverInExtracted.length() < 1000) {
                error("llama-server binary not found in extracted archive")
            }

            extractedDir.listFiles()?.forEach { f ->
                val target = File(dir, f.name)
                if (target.exists()) target.delete()
                f.copyTo(target, overwrite = true)
                if (f.name.endsWith(".so") || f.name == "llama-server" || f.name == "llama-cli") {
                    target.setExecutable(true, false)
                }
            }
            extractedDir.deleteRecursively()

            // Verify the source binary
            val source = File(dir, "llama-server")
            if (!source.exists()) error("llama-server not found after extraction")

            if (!verifyBinary(source, dir.absolutePath)) {
                error(
                    "Extracted llama-server is not a valid ELF binary. " +
                    "File: ${source.name} (${source.length()} bytes), " +
                    "Device: ${Build.MODEL} (${Build.HARDWARE})"
                )
            }

            // Deploy to nativeLibraryDir for execution
            val deployed = deployBinary(source)
            if (deployed == null) {
                error("Could not deploy llama-server to nativeLibraryDir for execution")
            }

            markInstalled(deployed)
            archiveFile.delete()
            return deployed

        } catch (e: Exception) {
            archiveFile.delete()
            throw e
        }
    }

    // ── GitHub API ──────────────────────────────────────────────────

    private fun latestReleaseTag(): String {
        try {
            val conn = URL("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
                .openConnection() as HttpURLConnection
            conn.setRequestProperty("User-Agent", "PhonoLabAndroid/1")
            conn.connectTimeout = 15_000
            conn.readTimeout = 15_000
            val status = conn.responseCode
            if (status != 200) {
                conn.disconnect()
                error("GitHub API returned HTTP $status")
            }
            val body = conn.inputStream.bufferedReader().readText()
            conn.disconnect()
            val tag = JSONObject(body).optString("tag_name", "")
            if (tag.isEmpty()) error("GitHub API returned empty tag_name")
            return tag
        } catch (e: Exception) {
            error("Could not fetch release info from GitHub: ${e.message}")
        }
    }

    private fun downloadFile(
        urlStr: String,
        dest: File,
        label: String,
        progress: (done: Long, total: Long, label: String) -> Unit,
    ) {
        dest.parentFile?.mkdirs()
        val part = File(dest.parentFile, dest.name + ".part")
        var currentUrl = urlStr
        var redirects = 0

        while (redirects < 10) {
            val conn = URL(currentUrl).openConnection() as HttpURLConnection
            conn.setRequestProperty("User-Agent", "PhonoLabAndroid/1")
            conn.connectTimeout = 30_000
            conn.readTimeout = 300_000
            conn.instanceFollowRedirects = false

            val status = conn.responseCode
            if (status in 301..308) {
                currentUrl = conn.getHeaderField("Location") ?: error("Redirect with no Location header")
                conn.disconnect()
                redirects++
                continue
            }
            if (status == 404) {
                conn.disconnect()
                error("Release asset not found (404).\nURL: $currentUrl")
            }
            if (status != 200) {
                val errBody = try { conn.errorStream?.bufferedReader()?.readText()?.take(500) ?: "" } catch (_: Exception) { "" }
                conn.disconnect()
                error("HTTP $status: $errBody\nURL: $currentUrl")
            }

            val total = conn.contentLengthLong.coerceAtLeast(0L)
            var done = 0L
            conn.inputStream.use { input ->
                FileOutputStream(part).use { output ->
                    val buffer = ByteArray(8192)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        done += read.toLong()
                        progress(done, total, label)
                    }
                }
            }
            conn.disconnect()
            if (dest.exists()) dest.delete()
            require(part.renameTo(dest)) { "Could not finalize download." }
            return
        }
        error("Too many redirects for $urlStr")
    }

    private fun extractTarGz(archive: File, destDir: File) {
        destDir.mkdirs()
        val proc = ProcessBuilder(
            "tar", "xzf", archive.absolutePath,
            "-C", destDir.absolutePath
        )
            .redirectErrorStream(true)
            .start()
        val output = proc.inputStream.bufferedReader().readText()
        val exitCode = proc.waitFor()
        if (exitCode != 0) {
            error("tar failed (exit $exitCode): ${output.take(500)}")
        }
    }

    private fun markInstalled(server: File) {
        prefs.edit()
            .putBoolean("installed", true)
            .putString("server_path", server.absolutePath)
            .putLong("installed_at", System.currentTimeMillis())
            .apply()
        store.markBinaryInstalled()
    }
}
