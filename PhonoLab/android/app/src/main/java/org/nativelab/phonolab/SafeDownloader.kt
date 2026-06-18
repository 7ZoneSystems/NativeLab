package org.nativelab.phonolab

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

class SafeDownloader(private val store: PhonoLabStore) {
    private val maxDownloadBytes = 7L * 1024L * 1024L * 1024L

    fun fetchGgufFiles(repo: String): List<RemoteFile> {
        require(Regex("^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$").matches(repo)) {
            "Invalid Hugging Face repo id"
        }
        val conn = open("https://huggingface.co/api/models/$repo")
        conn.connectTimeout = 30_000
        conn.readTimeout = 45_000
        conn.inputStream.bufferedReader().use { reader ->
            val json = JSONObject(reader.readText())
            val siblings = json.optJSONArray("siblings") ?: return emptyList()
            val out = mutableListOf<RemoteFile>()
            for (idx in 0 until siblings.length()) {
                val row = siblings.optJSONObject(idx) ?: continue
                val name = row.optString("rfilename", "")
                if (!name.lowercase().endsWith(".gguf")) continue
                val lfs = row.optJSONObject("lfs")
                val size = row.optLong("size", lfs?.optLong("size", 0L) ?: 0L)
                out.add(RemoteFile(name, size))
            }
            return out
        }
    }

    fun chooseFile(files: List<RemoteFile>, candidate: ModelCandidate): RemoteFile {
        fun score(file: RemoteFile): Int {
            val lower = file.name.lowercase()
            if (!lower.endsWith(".gguf")) return -1000
            if (listOf("mmproj", "projector", "clip").any { lower.contains(it) }) return -1000
            if (lower.contains("-of-") || lower.contains("00001-of-")) return -900
            val upper = file.name.uppercase()
            var value = 0
            candidate.quantPreferences.forEachIndexed { index, quant ->
                if (upper.contains(quant)) {
                    value += 120 - index * 8
                    return@forEachIndexed
                }
            }
            if (lower.contains("instruct") || lower.contains("chat")) value += 8
            if (lower.contains("imat") || lower.contains("imatrix")) value += 3
            return value
        }
        return files.maxWithOrNull(compareBy<RemoteFile> { score(it) }.thenBy { -it.size })
            ?.takeIf { score(it) > 0 }
            ?: error("No compatible GGUF file found for ${candidate.label}")
    }

    fun downloadModel(
        candidate: ModelCandidate,
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File {
        val selected = chooseFile(fetchGgufFiles(candidate.repo), candidate)
        val dest = store.safeChild(File(store.modelsDir, candidate.key), File(selected.name).name)

        // Skip download if identical file already exists
        if (dest.exists() && dest.length() == selected.size) {
            return dest
        }

        val url = "https://huggingface.co/${candidate.repo}/resolve/main/${encodePath(selected.name)}"
        download(url, dest, selected.size, progress)

        // Clean up duplicates: if other .gguf files exist for this candidate,
        // keep only the largest one
        cleanupDuplicates(dest)

        return dest
    }

    /**
     * Remove smaller duplicate .gguf files in the same directory.
     * Keeps only the largest file (the one just downloaded is likely the right quant).
     */
    private fun cleanupDuplicates(downloaded: File) {
        val parent = downloaded.parentFile ?: return
        if (!parent.exists()) return
        val ggufFiles = parent.listFiles()?.filter {
            it.isFile && it.extension.equals("gguf", ignoreCase = true) && it != downloaded
        } ?: return
        for (file in ggufFiles) {
            // Delete if smaller than the just-downloaded file, or if same name (stale .part)
            if (file.length() < downloaded.length()) {
                file.delete()
            }
        }
    }

    fun download(
        url: String,
        dest: File,
        expectedSize: Long = 0L,
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File {
        require(expectedSize <= 0L || expectedSize <= maxDownloadBytes) {
            "Download is larger than the mobile safety limit."
        }
        dest.parentFile?.mkdirs()
        val part = File(dest.parentFile, dest.name + ".part")
        var resumeFrom = if (part.exists()) part.length() else 0L
        val conn = open(url)
        if (resumeFrom > 0L) {
            conn.setRequestProperty("Range", "bytes=$resumeFrom-")
        }
        conn.connectTimeout = 30_000
        conn.readTimeout = 45_000
        val status = conn.responseCode
        if (resumeFrom > 0L && status != HttpURLConnection.HTTP_PARTIAL) {
            resumeFrom = 0L
        }
        val total = when {
            expectedSize > 0L -> expectedSize
            status == HttpURLConnection.HTTP_PARTIAL -> resumeFrom + conn.contentLengthLong.coerceAtLeast(0L)
            else -> conn.contentLengthLong.coerceAtLeast(0L)
        }
        require(total <= 0L || total <= maxDownloadBytes) {
            "Download is larger than the mobile safety limit."
        }
        val modeAppend = resumeFrom > 0L
        var done = resumeFrom
        conn.inputStream.use { input ->
            FileOutputStream(part, modeAppend).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    done += read.toLong()
                    require(done <= maxDownloadBytes) {
                        "Download exceeded the mobile safety limit."
                    }
                    progress(done, total, dest.name)
                }
            }
        }
        if (total > 0L && done < total) error("Download was incomplete.")
        if (dest.exists()) dest.delete()
        if (!part.renameTo(dest)) {
            // renameTo can fail on cross-filesystem moves; fallback to copy+delete
            part.inputStream().use { input ->
                dest.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            part.delete()
        }
        return dest
    }

    private fun open(url: String): HttpURLConnection {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.setRequestProperty("User-Agent", "PhonoLabAndroid/1")
        return conn
    }

    private fun encodePath(path: String): String {
        return path.split("/").joinToString("/") {
            URLEncoder.encode(it, "UTF-8").replace("+", "%20")
        }
    }
}
