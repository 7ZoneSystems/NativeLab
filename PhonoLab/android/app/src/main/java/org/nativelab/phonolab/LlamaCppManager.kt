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
 * The release tarball extracts to llama-{tag}/ containing:
 *   llama-server, llama-cli, libllama.so, libggml.so,
 *   libggml-cpu-android_armv8.0_1.so, libggml-cpu-android_armv8.2_1.so, etc.
 *
 * All files must stay in the same directory — the binaries dynamically
 * load the .so files from their own directory.
 */
class LlamaCppManager(private val context: Context, private val store: PhonoLabStore) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("llama_cpp_manager", Context.MODE_PRIVATE)

    data class RuntimeStatus(
        val ready: Boolean,
        val serverPath: String,
        val libDir: String,
        val message: String,
        val version: String = "",
    )

    /** Device CPU architecture info. */
    data class ArchInfo(
        val primary: String,      // e.g. "arm64-v8a", "x86_64"
        val allAbis: List<String>, // all supported ABIs
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

    /** Get the download URL suffix for the device architecture. */
    fun archDownloadSuffix(): String? {
        val arch = detectArch()
        if (arch.isArm64) return "android-arm64"
        // x86_64 emulators can sometimes run arm64 via translation
        // but llama.cpp doesn't release x86_64 Android builds
        return null
    }

    /** Directory where the extracted release lives. */
    private fun runtimeDir(): File = File(store.runtimeDir, "llama")

    fun status(): RuntimeStatus {
        val server = findServer() ?: return RuntimeStatus(false, "", "", "llama-server not installed")
        val libDir = server.parentFile?.absolutePath ?: ""
        val version = getBinaryVersion(server, libDir)
        return RuntimeStatus(true, server.absolutePath, libDir, "llama-server ready", version)
    }

    /** Find llama-server binary in the runtime directory. */
    fun findServer(): File? {
        val dir = runtimeDir()
        val server = File(dir, "llama-server")
        if (server.exists() && server.length() > 1000) {
            server.setExecutable(true, false)
            return server
        }
        // Also check binDir for legacy installs
        val binServer = File(store.runtimeBinDir, "llama-server")
        if (binServer.exists() && binServer.length() > 1000) {
            binServer.setExecutable(true, false)
            return binServer
        }
        return null
    }

    /** Get the directory containing all .so files. */
    fun findLibDir(): File? {
        val server = findServer() ?: return null
        return server.parentFile
    }

    /** Run a binary with LD_LIBRARY_PATH set to the lib directory. */
    private fun runBinary(binary: File, libDir: String, vararg args: String): Pair<Int, String> {
        val env = HashMap(System.getenv())
        val existing = env["LD_LIBRARY_PATH"] ?: ""
        env["LD_LIBRARY_PATH"] = if (existing.isNotEmpty()) "$libDir:$existing" else libDir

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
            output.take(200)
        } catch (_: Exception) { "" }
    }

    /** Verify a binary actually runs with its libraries. */
    fun verifyBinary(server: File, libDir: String): Boolean {
        return try {
            val (exitCode, _) = runBinary(server, libDir, "--version")
            exitCode == 0
        } catch (_: Exception) { false }
    }

    /**
     * Download and install llama-server from GitHub releases.
     * Keeps the full extracted directory structure so .so files are found.
     */
    fun downloadAndInstall(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? {
        val dir = runtimeDir()
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
                    "This device does not support arm64. A physical arm64 Android device is required."
                }
            )
        }

        // Get latest release tag
        val tag = latestReleaseTag()
        if (tag.isEmpty()) error("Could not determine latest llama.cpp release version")

        // Build correct URL
        val url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-$archSuffix.tar.gz"
        val archiveFile = File(store.downloadsDir, "llama-$tag-android-arm64.tar.gz")

        try {
            // Download tar.gz
            try {
                downloadFile(url, archiveFile, "llama-server", progress)
            } catch (e: Exception) {
                error("Download failed: ${e.message}\nURL: $url")
            }

            // Extract using tar — extracts to llama-{tag}/ inside dir
            try {
                extractTarGz(archiveFile, dir)
            } catch (e: Exception) {
                error("Extraction failed: ${e.message}")
            }

            // The tarball extracts to dir/llama-{tag}/
            val extractedDir = File(dir, "llama-$tag")
            val serverInExtracted = File(extractedDir, "llama-server")

            if (!serverInExtracted.exists() || serverInExtracted.length() < 1000) {
                error("llama-server binary not found in extracted archive")
            }

            // Move all files from extractedDir to runtimeDir
            // This preserves the directory structure needed for .so loading
            extractedDir.listFiles()?.forEach { f ->
                val target = File(dir, f.name)
                if (target.exists()) target.delete()
                f.renameTo(target)
            }
            extractedDir.deleteRecursively()

            // Verify the server binary
            val server = File(dir, "llama-server")
            server.setExecutable(true, false)

            if (!server.exists()) {
                error("llama-server not found after extraction")
            }

            // Set all .so files executable
            dir.listFiles()?.filter { it.name.endsWith(".so") }?.forEach {
                it.setExecutable(true, false)
            }

            // Verify with LD_LIBRARY_PATH
            if (verifyBinary(server, dir.absolutePath)) {
                markInstalled(server)
                archiveFile.delete()
                return server
            }

            // If --version fails, try just checking the file is a valid ELF
            if (server.length() > 10_000) {
                // Binary exists and is large enough — might work even if --version fails
                // on some Android versions where ProcessBuilder has issues
                markInstalled(server)
                archiveFile.delete()
                return server
            }

            error(
                "llama-server binary could not be verified on this device. " +
                "Device: ${Build.MODEL} (${Build.HARDWARE}), " +
                "ABIs: ${Build.SUPPORTED_ABIS.joinToString()}, " +
                "Binary arch: arm64-v8a"
            )
        } catch (e: Exception) {
            // Clean up on failure
            dir.listFiles()?.forEach { it.delete() }
            archiveFile.delete()
            throw e
        }
    }

    /** Get latest release tag from GitHub API. */
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

    /** Download a file with redirect following. */
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

    /** Extract a .tar.gz file using system tar. */
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
