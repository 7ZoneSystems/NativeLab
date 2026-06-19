package org.nativelab.phonolab

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import org.nativelab.phonolab.data.ModelManager
import org.nativelab.phonolab.data.SessionManager
import org.nativelab.phonolab.ui.ErrorBannerView
import kotlin.system.exitProcess

/**
 * Application-level singletons + global crash handler.
 * Survives activity recreation (theme switch, config change).
 */
class PhonoLabApp : Application() {

    lateinit var store: PhonoLabStore
        private set
    lateinit var sessionManager: SessionManager
        private set
    lateinit var modelManager: ModelManager
        private set
    lateinit var runtime: LlamaRuntime
        private set

    var currentActivity: Activity? = null
        private set

    override fun onCreate() {
        super.onCreate()

        // Install global crash handler BEFORE anything else
        installCrashHandler()

        store = PhonoLabStore(this)
        sessionManager = SessionManager(store)
        modelManager = ModelManager(store)
        runtime = LlamaRuntime(this, store, modelManager = modelManager)

        // Track current activity for crash recovery
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityCreated(a: Activity, b: Bundle?) { currentActivity = a }
            override fun onActivityStarted(a: Activity) {}
            override fun onActivityResumed(a: Activity) { currentActivity = a }
            override fun onActivityPaused(a: Activity) {}
            override fun onActivityStopped(a: Activity) {}
            override fun onActivitySaveInstanceState(a: Activity, b: Bundle) {}
            override fun onActivityDestroyed(a: Activity) {
                if (currentActivity === a) currentActivity = null
            }
        })
    }

    /**
     * Global crash handler - catches any uncaught exception.
     * Logs it, cleans up server processes, and shows a restart dialog.
     * Does NOT force-kill the app (lets Android handle lifecycle).
     */
    private fun installCrashHandler() {
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                Log.e(TAG, "Uncaught exception on ${thread.name}", throwable)

                // Emergency cleanup: kill all llama processes
                try {
                    if (::runtime.isInitialized) {
                        runtime.killAllLlamaProcesses()
                    }
                } catch (_: Exception) { }

                // Build error message based on exception type
                val (title, message) = when (throwable) {
                    is IllegalStateException -> {
                        "Illegal State" to buildString {
                            append("An internal state error occurred.\n\n")
                            append("Error: ${throwable.message?.take(200)}\n\n")
                            append("This usually means a component was used after ")
                            append("being cleaned up or in the wrong order.\n")
                            append("Restarting the app should fix this.")
                        }
                    }
                    is IllegalArgumentException -> {
                        "Invalid Argument" to buildString {
                            append("An invalid operation was attempted.\n\n")
                            append("Error: ${throwable.message?.take(200)}")
                        }
                    }
                    is NullPointerException -> {
                        "Null Reference" to buildString {
                            append("A required component was missing.\n\n")
                            append("Error: ${throwable.message?.take(200)}\n\n")
                            append("This may happen if the app was interrupted during loading.")
                        }
                    }
                    is OutOfMemoryError -> {
                        "Out of Memory" to buildString {
                            append("The device ran out of memory.\n\n")
                            append("Try closing other apps or using a smaller model.")
                        }
                    }
                    else -> {
                        "Fatal Error" to "PhonoLab encountered a fatal error:\n\n${throwable.message?.take(200)}"
                    }
                }

                // Show fatal error dialog on main thread if possible
                try {
                    val activity = currentActivity
                    if (activity != null) {
                        activity.runOnUiThread {
                            try {
                                android.app.AlertDialog.Builder(activity)
                                    .setTitle(title)
                                    .setMessage(message)
                                    .setCancelable(false)
                                    .setPositiveButton("Restart") { _, _ ->
                                        activity.recreate()
                                    }
                                    .setNegativeButton("Exit") { _, _ ->
                                        activity.finish()
                                    }
                                    .show()
                            } catch (_: Exception) {
                                // Fallback to toast if dialog fails
                                Toast.makeText(
                                    activity,
                                    "PhonoLab crashed: ${throwable.message?.take(100)}",
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                        }
                        // Give dialog time to show
                        Thread.sleep(2000)
                    }
                } catch (_: Exception) { }

            } catch (_: Exception) {
                // If our handler itself fails, fall through to default
            } finally {
                // Delegate to default handler (shows system crash dialog or kills process)
                defaultHandler?.uncaughtException(thread, throwable)
            }
        }
    }

    /**
     * Show an error to the user.
     * - fatal=true: shows a blocking dialog with Restart/Exit buttons
     * - fatal=false: shows a red exclamation banner that auto-dismisses
     */
    fun showError(message: String, fatal: Boolean = false) {
        Log.e(TAG, "showError(fatal=$fatal): $message")
        val activity = currentActivity ?: run {
            Log.w(TAG, "No activity to show error: $message")
            return
        }

        if (fatal) {
            activity.runOnUiThread {
                try {
                    android.app.AlertDialog.Builder(activity)
                        .setTitle("Error")
                        .setMessage(message)
                        .setCancelable(true)
                        .setPositiveButton("OK", null)
                        .show()
                } catch (_: Exception) {
                    Toast.makeText(activity, message, Toast.LENGTH_LONG).show()
                }
            }
        } else {
            activity.runOnUiThread {
                showBanner(activity, message)
            }
        }
    }

    private fun showBanner(activity: Activity, message: String) {
        try {
            val decor = activity.window.decorView as ViewGroup
            // Find or create banner
            var banner = decor.findViewWithTag<ErrorBannerView>(BANNER_TAG)
            if (banner == null) {
                banner = ErrorBannerView(activity).apply { tag = BANNER_TAG }
                decor.addView(banner, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    val dp = (16 * activity.resources.displayMetrics.density).toInt()
                    setMargins(dp, dp, dp, 0)
                })
            }
            banner.show(message)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show banner", e)
            Toast.makeText(activity, message, Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Safe execute - runs a block and catches any exception.
     * Returns null on failure, logs the error.
     */
    fun <T> safeRun(tag: String, block: () -> T): T? {
        return try {
            block()
        } catch (e: Exception) {
            Log.e(tag, "safeRun failed", e)
            null
        }
    }

    /**
     * Safe execute specifically for operations that may throw IllegalStateException.
     * Catches IllegalStateException, shows a non-fatal banner to the user, and returns null.
     * Use this for operations like: check(), require(), state-dependent calls.
     *
     * @param tag Log tag
     * @param operation Description of what was being attempted (shown to user)
     * @param block The code to execute
     * @return Result or null if IllegalStateException was caught
     */
    fun <T> safeRunState(tag: String, operation: String, block: () -> T): T? {
        return try {
            block()
        } catch (e: IllegalStateException) {
            Log.e(tag, "IllegalStateException during $operation", e)
            showError("$operation failed: ${e.message?.take(150)}", fatal = false)
            null
        } catch (e: IllegalArgumentException) {
            Log.e(tag, "IllegalArgumentException during $operation", e)
            showError("$operation failed: ${e.message?.take(150)}", fatal = false)
            null
        } catch (e: Exception) {
            Log.e(tag, "Unexpected error during $operation", e)
            showError("$operation failed unexpectedly", fatal = false)
            null
        }
    }

    companion object {
        private const val TAG = "PhonoLabApp"
        private const val BANNER_TAG = "error_banner_view"
    }
}
