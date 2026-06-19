package org.nativelab.phonolab

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Manages llama-server binary bundled in the APK.
 *
 * Architecture:
 *   APK jniLibs/arm64-v8a/libllama_server.so  (stub launcher)
 *   APK jniLibs/arm64-v8a/libllama-server-impl.so  (real server)
 *   APK jniLibs/arm64-v8a/libllama.so, libggml.so, etc.
 *
 * At install time, Android extracts ALL lib*.so to nativeLibraryDir:
 *   /data/app/<pkg>/lib/arm64/
 *
 * At runtime, exec(nativeLibraryDir/libllama_server.so) via JNI fork+execve.
 * No runtime downloads, no permission issues, no W^X violations.
 */
class LlamaCppManager(
    private val context: Context,
    private val store: PhonoLabStore,
    private val storageManager: StorageManager? = null,
) {

    companion object {
        private const val TAG = "LlamaCppManager"
        private const val SERVER_BINARY = "libllama_server.so"

        var nativeLoaded = false
            private set

        init {
            try {
                System.loadLibrary("runner")
                nativeLoaded = true
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load librunner.so", e)
            }
        }

        /** Validate binary is a real ELF file. */
        fun isValidElf(file: File): Boolean {
            if (!file.exists() || file.length() < 4) return false
            return try {
                val header = ByteArray(4)
                file.inputStream().use { it.read(header) }
                header[0] == 0x7F.toByte() &&
                    header[1] == 'E'.code.toByte() &&
                    header[2] == 'L'.code.toByte() &&
                    header[3] == 'F'.code.toByte()
            } catch (e: Exception) { false }
        }
    }

    // JNI methods from runner.cpp
    external fun nativeExec(binary: String, args: Array<String>, env: Array<String>): Int
    external fun nativeKill(pid: Int): Boolean
    external fun nativeKillForcibly(pid: Int): Boolean
    external fun nativeWaitPid(pid: Int): Int

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
        if (arch.isArm64) return "arm64"
        return null
    }

    // ── Status ──────────────────────────────────────────────────────

    fun status(): RuntimeStatus {
        val server = findServer()
        if (server == null) {
            val arch = detectArch()
            return RuntimeStatus(
                false, "", "",
                "llama-server not bundled. Run setup_binaries.sh and rebuild APK.\n" +
                    "Device: ${arch.primary} (${arch.allAbis.joinToString()})"
            )
        }
        val libDir = nativeLibraryDir()
        val version = getBinaryVersion(server)
        return RuntimeStatus(true, server.absolutePath, libDir, "llama-server ready", version)
    }

    // ── Find binary ─────────────────────────────────────────────────

    /** nativeLibraryDir - the only directory Android allows exec() from. */
    private fun nativeLibraryDir(): String = context.applicationInfo.nativeLibraryDir

    /**
     * Find llama-server binary in nativeLibraryDir.
     * Bundled in APK as libllama_server.so, extracted by package manager at install.
     */
    fun findServer(): File? {
        val nativeDir = nativeLibraryDir()
        val bundled = File(nativeDir, SERVER_BINARY)

        if (bundled.exists() && isValidElf(bundled)) {
            Log.d(TAG, "Found bundled server: ${bundled.absolutePath} (${bundled.length()} bytes)")
            return bundled
        }

        // List what's actually in nativeLibraryDir for debugging
        val contents = File(nativeDir).listFiles()?.map { "${it.name} (${it.length()} bytes)" }
        Log.e(TAG, "libllama_server.so not found in nativeLibraryDir=$nativeDir, contents=$contents")
        return null
    }

    /**
     * LD_LIBRARY_PATH - only nativeLibraryDir needed.
     * All .so files live there; the stub dlopen()s impl from the same dir.
     */
    fun buildLdLibraryPath(): String = nativeLibraryDir()

    // ── Server management via JNI ───────────────────────────────────

    /**
     * Start llama-server via JNI fork()+execve().
     * Returns child PID, or -1 on failure.
     */
    fun startServer(
        serverBin: File,
        modelPath: File,
        port: Int,
        threads: Int = 4,
        ctxSize: Int = 2048,
        nPredict: Int = 384,
        mmproj: String? = null,
    ): Int {
        val nativeDir = nativeLibraryDir()

        if (!nativeLoaded) {
            Log.e(TAG, "Cannot start server: librunner.so not loaded")
            return -1
        }
        if (!serverBin.exists()) {
            Log.e(TAG, "Server binary not found: ${serverBin.absolutePath}")
            return -1
        }
        if (!modelPath.exists()) {
            Log.e(TAG, "Model not found: ${modelPath.absolutePath}")
            return -1
        }

        Log.d(TAG, "nativeDir: $nativeDir")
        Log.d(TAG, "binary: ${serverBin.absolutePath} (${serverBin.length()} bytes)")
        Log.d(TAG, "model: ${modelPath.absolutePath} (${modelPath.length()} bytes)")

        // Args - do NOT put binary path here, runner.cpp prepends it as argv[0]
        val args = mutableListOf(
            "-m", modelPath.absolutePath,
            "--port", port.toString(),
            "-t", threads.toString(),
            "--ctx-size", ctxSize.toString(),
            "-n", nPredict.toString(),
            "--host", "127.0.0.1",
        )
        if (!mmproj.isNullOrEmpty()) {
            args.add("--mmproj")
            args.add(mmproj)
        }

        // env - LD_LIBRARY_PATH MUST be nativeLibraryDir and nothing else
        // All .so files live there; the stub dlopen()s impl from the same dir
        val env = arrayOf(
            "LD_LIBRARY_PATH=$nativeDir",
            "HOME=${context.filesDir.absolutePath}",
            "TMPDIR=${context.cacheDir.absolutePath}",
            "PATH=/system/bin:/system/xbin",
        )

        Log.d(TAG, "Starting with LD_LIBRARY_PATH=$nativeDir")

        val pid = try {
            nativeExec(serverBin.absolutePath, args.toTypedArray(), env)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "JNI not loaded - runner.so missing?", e)
            return -1
        } catch (e: Exception) {
            Log.e(TAG, "nativeExec failed", e)
            return -1
        }
        if (pid <= 0) {
            Log.e(TAG, "nativeExec returned invalid pid=$pid")
            return -1
        }

        Log.d(TAG, "Started PID: $pid")
        return pid
    }

    /** Kill server process gracefully. */
    fun killServer(pid: Int) {
        if (pid <= 0 || !nativeLoaded) return
        try {
            Log.d(TAG, "Killing PID: $pid")
            nativeKill(pid)
            Thread.sleep(500)
            if (nativeWaitPid(pid) == 0) {
                Log.d(TAG, "Force killing PID: $pid")
                nativeKillForcibly(pid)
            }
        } catch (e: Exception) {
            Log.e(TAG, "killServer failed for pid=$pid", e)
        }
    }

    /** Check if process is still running. */
    fun isRunning(pid: Int): Boolean {
        if (pid <= 0 || !nativeLoaded) return false
        return try { nativeWaitPid(pid) == 0 } catch (_: Exception) { false }
    }

    /** Get process exit code (0 = still running, >0 = exited). */
    fun checkProcess(pid: Int): Int {
        if (pid <= 0 || !nativeLoaded) return -1
        return try { nativeWaitPid(pid) } catch (_: Exception) { -1 }
    }

    // ── Version check ───────────────────────────────────────────────

    private fun getBinaryVersion(server: File): String {
        if (!nativeLoaded) return "native-not-loaded"
        return try {
            val env = arrayOf("LD_LIBRARY_PATH=${nativeLibraryDir()}")
            val pid = nativeExec(server.absolutePath, arrayOf("--version"), env)
            if (pid <= 0) return "unknown"
            // Wait for process to finish
            var exitCode = 0
            for (i in 0..50) {
                exitCode = nativeWaitPid(pid)
                if (exitCode != 0) break
                Thread.sleep(100)
            }
            "exit=$exitCode"
        } catch (e: Exception) {
            Log.w(TAG, "Version check failed", e)
            "unknown"
        }
    }

    // ── Download & Install (legacy - for runtime download fallback) ──

    fun downloadAndInstall(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? {
        return try {
            doDownloadAndInstall(progress)
        } catch (e: Exception) {
            Log.e(TAG, "downloadAndInstall failed", e)
            null
        }
    }

    private fun doDownloadAndInstall(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? {
        // With bundled binaries, this should not be needed.
        // But keep as fallback for development/testing.
        val dir = store.runtimeDir
        dir.mkdirs()

        val arch = detectArch()
        if (!arch.isArm64) {
            error(
                "This device runs ${arch.primary} (${arch.allAbis.joinToString()}). " +
                    "llama.cpp only provides arm64-v8a Android binaries."
            )
        }

        val tag = latestReleaseTag()
        if (tag.isEmpty()) error("Could not determine latest llama.cpp release version")

        val suffix = archDownloadSuffix() ?: error("No llama.cpp build for ${arch.primary}")
        val url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-android-$suffix.tar.gz"
        val archiveFile = File(store.downloadsDir, "llama-$tag-android.tar.gz")

        try {
            // Validate URL exists before downloading
            validateUrl(url)

            downloadFile(url, archiveFile, "llama-server", progress)
            // Extract and deploy to filesDir (for exec via JNI)
            extractTarGz(archiveFile, dir)
            val source = File(dir, "llama-server")
            if (!source.exists()) error("llama-server not found in archive")
            val dest = File(context.filesDir, "llama-server")
            copyFile(source, dest)
            dest.setExecutable(true, false)
            // Copy .so deps
            dir.listFiles()?.filter { it.name.endsWith(".so") }?.forEach { so ->
                copyFile(so, File(context.filesDir, so.name))
            }
            archiveFile.delete()
            return dest
        } catch (e: Exception) {
            archiveFile.delete()
            throw e
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    private fun copyFile(source: File, target: File) {
        target.parentFile?.mkdirs()
        if (target.exists()) target.delete()
        source.inputStream().use { input ->
            target.outputStream().use { output ->
                val buffer = ByteArray(8192)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
    }

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
                Log.e(TAG, "GitHub API returned HTTP $status")
                return ""
            }
            val body = conn.inputStream.bufferedReader().readText()
            conn.disconnect()
            val tag = JSONObject(body).optString("tag_name", "")
            if (tag.isEmpty()) {
                Log.e(TAG, "GitHub API returned empty tag_name")
                return ""
            }
            return tag
        } catch (e: Exception) {
            Log.e(TAG, "Could not fetch release info", e)
            return ""
        }
    }

    /** HEAD request to validate URL exists before downloading. */
    private fun validateUrl(urlStr: String) {
        var currentUrl = urlStr
        var redirects = 0
        while (redirects < 5) {
            val conn = URL(currentUrl).openConnection() as HttpURLConnection
            conn.requestMethod = "HEAD"
            conn.setRequestProperty("User-Agent", "PhonoLabAndroid/1")
            conn.connectTimeout = 15_000
            conn.readTimeout = 15_000
            conn.instanceFollowRedirects = false
            try {
                val status = conn.responseCode
                if (status in 301..308) {
                    currentUrl = conn.getHeaderField("Location") ?: error("Redirect with no Location")
                    redirects++
                    continue
                }
                if (status == 404) {
                    error(
                        "Release asset not found (404).\n" +
                            "URL: $currentUrl\n" +
                            "The release may not have an Android binary for this architecture.\n" +
                            "Check: https://github.com/ggml-org/llama.cpp/releases/latest"
                    )
                }
                if (status != 200) {
                    error("URL check failed: HTTP $status for $currentUrl")
                }
                return  // URL is valid
            } finally {
                conn.disconnect()
            }
        }
        error("Too many redirects checking URL: $urlStr")
    }

    private fun downloadFile(
        urlStr: String, dest: File, label: String,
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
                currentUrl = conn.getHeaderField("Location") ?: error("Redirect with no Location")
                conn.disconnect(); redirects++; continue
            }
            if (status != 200) {
                conn.disconnect(); error("HTTP $status for $currentUrl")
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
            if (!part.renameTo(dest)) {
                Log.w(TAG, "renameTo failed, falling back to copy+delete")
                copyFile(part, dest)
                part.delete()
            }
            return
        }
        error("Too many redirects")
    }

    private fun extractTarGz(archive: File, destDir: File) {
        destDir.mkdirs()
        val proc = ProcessBuilder("tar", "xzf", archive.absolutePath, "-C", destDir.absolutePath)
            .redirectErrorStream(true).start()
        val output = proc.inputStream.bufferedReader().readText()
        val exitCode = proc.waitFor()
        if (exitCode != 0) {
            error("tar failed (exit $exitCode): ${output.take(500)}")
        }
    }
}
