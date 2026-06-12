package org.nativelab.phonolab

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

class LlamaRuntime(
    private val context: Context,
    private val store: PhonoLabStore,
) {
    private val downloader = SafeDownloader(store)
    private var loadedModel: File? = null

    companion object {
        // Pre-built binary from ggml releases (Android arm64)
        private const val BINARY_RELEASE_URL =
            "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-cli-linux-android-arm64"
        private const val BINARY_ZIP_URL =
            "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-cli-android-arm64.zip"
    }

    fun pullLlamaSource(progress: (done: Long, total: Long, label: String) -> Unit): File {
        val archive = store.safeChild(store.downloadsDir, "llama.cpp-master.zip")
        downloader.download(
            "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip",
            archive,
            progress = progress,
        )
        unzipSafe(archive, store.sourceDir)
        return store.sourceDir
    }

    /**
     * Auto-install llama-cli binary for Android arm64.
     * Tries direct download first, then falls back to zip.
     * Returns the installed binary path, or null if download fails.
     */
    fun autoInstallBinary(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? {
        // Already installed?
        findCli()?.let { return it }

        val dest = File(store.runtimeBinDir, "llama-cli")
        dest.parentFile?.mkdirs()

        // Try direct binary download
        try {
            downloadBinary(BINARY_RELEASE_URL, dest, progress)
            dest.setExecutable(true, false)
            if (dest.exists() && dest.length() > 0) {
                store.markBinaryInstalled()
                return dest
            }
        } catch (_: Exception) {
            // Fall through to zip attempt
        }

        // Try zip download
        try {
            val zipFile = File(store.downloadsDir, "llama-cli-android-arm64.zip")
            downloader.download(BINARY_ZIP_URL, zipFile, progress = progress)
            extractBinaryFromZip(zipFile, dest)
            dest.setExecutable(true, false)
            if (dest.exists() && dest.length() > 0) {
                store.markBinaryInstalled()
                return dest
            }
        } catch (_: Exception) {
            // Fall through
        }

        return null
    }

    private fun downloadBinary(
        urlStr: String,
        dest: File,
        progress: (done: Long, total: Long, label: String) -> Unit,
    ) {
        dest.parentFile?.mkdirs()
        val part = File(dest.parentFile, dest.name + ".part")
        val conn = URL(urlStr).openConnection() as HttpURLConnection
        conn.setRequestProperty("User-Agent", "PhonoLabAndroid/1")
        conn.connectTimeout = 30_000
        conn.readTimeout = 60_000
        conn.instanceFollowRedirects = true

        val status = conn.responseCode
        // Follow redirect
        if (status in 301..308) {
            val redirect = conn.getHeaderField("Location") ?: error("Redirect with no Location")
            conn.disconnect()
            downloadBinary(redirect, dest, progress)
            return
        }
        require(status == 200) { "HTTP $status" }

        val total = conn.contentLengthLong.coerceAtLeast(0L)
        var done = 0L
        conn.inputStream.use { input ->
            FileOutputStream(part).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    done += read.toLong()
                    progress(done, total, "llama-cli")
                }
            }
        }
        if (dest.exists()) dest.delete()
        require(part.renameTo(dest)) { "Could not finalize binary download." }
    }

    private fun extractBinaryFromZip(archive: File, dest: File) {
        dest.parentFile?.mkdirs()
        ZipInputStream(archive.inputStream().buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.isDirectory) continue
                val name = entry.name.substringAfterLast("/")
                if (name == "llama-cli" || name.startsWith("llama-cli")) {
                    FileOutputStream(dest).use { output -> zip.copyTo(output) }
                    zip.closeEntry()
                    return
                }
                zip.closeEntry()
            }
        }
        error("No llama-cli binary found in zip archive")
    }

    fun installBundledRuntimeIfAvailable(): File? {
        val dest = File(store.runtimeBinDir, "llama-cli")
        if (dest.exists() && dest.canExecute()) return dest
        return try {
            context.assets.open("runtimes/android-arm64/llama-cli").use { input ->
                dest.parentFile?.mkdirs()
                FileOutputStream(dest).use { output -> input.copyTo(output) }
            }
            dest.setExecutable(true, false)
            dest
        } catch (_: Exception) {
            null
        }
    }

    fun findCli(): File? {
        // 1. Native library (most reliable on Android)
        val nativeCli = File(context.applicationInfo.nativeLibraryDir, "libllama-cli.so")
        if (nativeCli.exists()) {
            nativeCli.setExecutable(true, false)
            return nativeCli
        }
        // 2. Bundled asset
        val bundled = installBundledRuntimeIfAvailable()
        if (bundled != null && bundled.exists()) return bundled
        // 3. App-private installed binary
        val privateCli = File(store.runtimeBinDir, "llama-cli")
        if (privateCli.exists()) {
            privateCli.setExecutable(true, false)
            return privateCli
        }
        return null
    }

    fun load(model: File) {
        require(model.exists() && model.extension.equals("gguf", ignoreCase = true)) {
            "Choose a downloaded GGUF model first."
        }
        loadedModel = model
    }

    fun unload() {
        loadedModel = null
    }

    fun isModelLoaded(): Boolean = loadedModel != null

    fun generate(prompt: String, onToken: (String) -> Unit): String {
        val model = loadedModel ?: error("Load a model first.")
        val cli = findCli() ?: error(
            "llama-cli is missing. Run Setup to install the runtime.",
        )
        require(prompt.length <= 18_000) { "Prompt is too large for the mobile safety limit." }
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        val command = listOf(
            cli.absolutePath,
            "-m",
            model.absolutePath,
            "-t",
            threads.toString(),
            "--ctx-size",
            "2048",
            "-n",
            "384",
            "--no-display-prompt",
            "--no-escape",
            "-p",
            prompt,
        )
        val process = ProcessBuilder(command)
            .redirectErrorStream(true)
            .start()
        val out = StringBuilder()
        process.inputStream.bufferedReader().use { reader ->
            val buffer = CharArray(1)
            while (true) {
                val read = reader.read(buffer)
                if (read < 0) break
                val text = String(buffer, 0, read)
                out.append(text)
                onToken(text)
            }
        }
        val code = process.waitFor()
        require(code == 0) { "llama-cli exited with code $code" }
        return out.toString()
    }

    private fun unzipSafe(archive: File, dest: File) {
        val tmp = File(dest.parentFile, dest.name + ".tmp")
        if (tmp.exists()) tmp.deleteRecursively()
        tmp.mkdirs()
        ZipInputStream(archive.inputStream().buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.isDirectory) continue
                val parts = entry.name.split("/").filter { it.isNotBlank() }
                if (parts.size <= 1) continue
                val stripped = parts.drop(1).joinToString("/")
                val out = store.safeChild(tmp, stripped)
                out.parentFile?.mkdirs()
                FileOutputStream(out).use { output -> zip.copyTo(output) }
                zip.closeEntry()
            }
        }
        if (dest.exists()) dest.deleteRecursively()
        require(tmp.renameTo(dest)) { "Could not finalize llama.cpp source extraction." }
    }
}
