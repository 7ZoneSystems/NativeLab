package org.nativelab.phonolab

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.os.Environment
import android.provider.DocumentsContract
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import java.io.File

/**
 * Manages app storage using SAF (Storage Access Framework).
 * On first launch, asks user to pick a folder.
 * All data (models, runtime, sessions) goes into that folder.
 *
 * Folder structure:
 *   <UserFolder>/
 *     PhonoLab/
 *       llama-server/   ← binary + .so files
 *       models/          ← downloaded GGUF models
 *       sessions/        ← chat history JSON
 *       config/          ← app config
 *       downloads/       ← temp downloads
 */
class StorageManager(private val context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("phonolab_storage", Context.MODE_PRIVATE)

    /** The persisted URI for the user-selected folder. */
    var folderUri: Uri? = null
        private set

    /** Whether a folder has been selected. */
    val hasFolder: Boolean
        get() = folderUri != null && getBaseDir() != null

    /** Register the SAF launcher. Call in onCreate before any other init. */
    fun registerLauncher(activity: AppCompatActivity): ActivityResultLauncher<Uri?> {
        return activity.registerForActivityResult(
            ActivityResultContracts.OpenDocumentTree()
        ) { uri ->
            if (uri != null) {
                // Persist permission
                try {
                    context.contentResolver.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                    )
                } catch (_: Exception) {}

                // Save URI
                folderUri = uri
                prefs.edit().putString("folder_uri", uri.toString()).apply()

                // Create subdirectories
                createSubdirectories(uri)

                // Notify callback
                onFolderSelected?.invoke(uri)
            }
        }
    }

    /** Callback when folder is selected. */
    var onFolderSelected: ((Uri) -> Unit)? = null

    /** Load persisted folder URI. Returns true if found. */
    fun loadPersistedUri(): Boolean {
        val uriStr = prefs.getString("folder_uri", null) ?: return false
        return try {
            val uri = Uri.parse(uriStr)
            // Check we still have permission
            val persisted = context.contentResolver.persistedUriPermissions
            val hasPerm = persisted.any { it.uri == uri && it.isReadPermission && it.isWritePermission }
            if (hasPerm) {
                folderUri = uri
                true
            } else {
                false
            }
        } catch (_: Exception) {
            false
        }
    }

    /** Launch the folder picker. */
    fun pickFolder(launcher: ActivityResultLauncher<Uri?>) {
        // Suggest a default location
        val initialUri = Uri.parse("content://com.android.externalstorage.documents/document/primary%3A")
        launcher.launch(initialUri)
    }

    /** Get the base PhonoLab directory as a real File path (for binary execution). */
    fun getBaseDir(): File? {
        val uri = folderUri ?: return null
        // Try to get real path from URI
        val path = getRealPathFromUri(uri)
        if (path != null) {
            val dir = File(path, "PhonoLab")
            dir.mkdirs()
            return dir
        }
        // Fallback: use app-private storage
        return File(context.filesDir, "PhonoLab").also { it.mkdirs() }
    }

    /** Get the llama-server directory (binary + .so files). */
    fun getLlamaServerDir(): File {
        val base = getBaseDir() ?: File(context.filesDir, "PhonoLab")
        return File(base, "llama-server").also { it.mkdirs() }
    }

    /** Get the models directory. */
    fun getModelsDir(): File {
        val base = getBaseDir() ?: File(context.filesDir, "PhonoLab")
        return File(base, "models").also { it.mkdirs() }
    }

    /** Get the sessions directory. */
    fun getSessionsDir(): File {
        val base = getBaseDir() ?: File(context.filesDir, "PhonoLab")
        return File(base, "sessions").also { it.mkdirs() }
    }

    /** Get the config directory. */
    fun getConfigDir(): File {
        val base = getBaseDir() ?: File(context.filesDir, "PhonoLab")
        return File(base, "config").also { it.mkdirs() }
    }

    /** Get the downloads directory. */
    fun getDownloadsDir(): File {
        val base = getBaseDir() ?: File(context.filesDir, "PhonoLab")
        return File(base, "downloads").also { it.mkdirs() }
    }

    /** Get the path shown to the user. */
    fun getDisplayPath(): String {
        val uri = folderUri ?: return "Not set"
        val path = getRealPathFromUri(uri)
        return path ?: uri.lastPathSegment ?: "Unknown"
    }

    /** Create subdirectories in the selected folder. */
    private fun createSubdirectories(uri: Uri) {
        val docFile = DocumentFile.fromTreeUri(context, uri) ?: return
        val subdirs = listOf("llama-server", "models", "sessions", "config", "downloads")
        for (name in subdirs) {
            if (docFile.findFile(name) == null) {
                docFile.createDirectory(name)
            }
        }
    }

    /** Try to get a real filesystem path from a content URI. */
    private fun getRealPathFromUri(uri: Uri): String? {
        try {
            // Handle external storage documents
            val docId = DocumentsContract.getTreeDocumentId(uri)
            if (docId.startsWith("primary:")) {
                val path = docId.removePrefix("primary:")
                val external = Environment.getExternalStorageDirectory().absolutePath
                return "$external/$path"
            }
            // Handle raw paths
            if (docId.contains(":")) {
                val parts = docId.split(":")
                if (parts.size >= 2) {
                    val type = parts[0]
                    val path = parts[1]
                    when (type) {
                        "primary" -> {
                            val external = Environment.getExternalStorageDirectory().absolutePath
                            return "$external/$path"
                        }
                        else -> {
                            // Try /storage/type/path
                            val storagePath = "/storage/$type/$path"
                            if (File(storagePath).exists()) return storagePath
                        }
                    }
                }
            }
        } catch (_: Exception) {}
        return null
    }
}
