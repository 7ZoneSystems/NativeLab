package org.nativelab.phonolab

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.adapter.SessionAdapter
import org.nativelab.phonolab.data.ChatSession
import org.nativelab.phonolab.data.ModelManager
import org.nativelab.phonolab.data.SessionManager
import org.nativelab.phonolab.theme.ThemeManager
import org.nativelab.phonolab.ui.ApiFragment
import org.nativelab.phonolab.ui.ChatFragment
import org.nativelab.phonolab.ui.DownloadsFragment
import org.nativelab.phonolab.ui.ModelsFragment

class MainActivity : AppCompatActivity(), ChatFragment.Host {

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var toolbar: MaterialToolbar
    private lateinit var sessionAdapter: SessionAdapter
    private lateinit var sessionManager: SessionManager
    private lateinit var store: PhonoLabStore
    private lateinit var runtime: LlamaRuntime
    private lateinit var storageManager: StorageManager

    private var sessions = listOf<ChatSession>()
    private var activeSessionId = ""
    private var currentFragmentTag = "chat" // chat, models, downloads, api
    @Volatile var isSelectingImage = false

    // SAF folder picker launcher
    private lateinit var folderPickerLauncher: ActivityResultLauncher<Uri?>

    // Permission request launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val denied = results.filter { !it.value }.keys
        if (denied.isNotEmpty()) {
            val names = denied.map { it.substringAfterLast(".") }.joinToString()
            Toast.makeText(this, "Denied: $names - some features may not work", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Edge-to-edge for Android 15+ (API 35+)
        if (Build.VERSION.SDK_INT >= 35) {
            enableEdgeToEdge()
        }
        ThemeManager.init(this)
        ThemeManager.applyTheme()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize storage manager
        storageManager = StorageManager(this)
        folderPickerLauncher = storageManager.registerLauncher(this)
        storageManager.onFolderSelected = { uri -> onFolderReady(uri) }

        // Check if folder is already selected (or default was chosen)
        if (storageManager.loadPersistedUri()) {
            initializeApp(savedInstanceState)
        } else {
            showFolderPickerDialog()
        }
    }

    private fun showFolderPickerDialog() {
        android.app.AlertDialog.Builder(this)
            .setTitle("Choose Storage Location")
            .setMessage(
                "PhonoLab needs a folder to store:\n\n" +
                "• llama-server runtime\n" +
                "• Downloaded models\n" +
                "• Chat history & settings\n\n" +
                "Select a folder on your device."
            )
            .setCancelable(false)
            .setPositiveButton("Choose Folder") { _, _ ->
                storageManager.pickFolder(folderPickerLauncher)
            }
            .setNegativeButton("Use Default") { _, _ ->
                storageManager.markUsingDefault()
                onFolderReady(null)
            }
            .show()
    }

    private fun onFolderReady(uri: Uri?) {
        initializeApp(null)
        Toast.makeText(this, "Storage: ${storageManager.getDisplayPath()}", Toast.LENGTH_LONG).show()
    }

    private fun initializeApp(savedInstanceState: Bundle?) {
        // Use application-level singletons (survive recreate)
        val app = application as PhonoLabApp
        store = app.store
        sessionManager = app.sessionManager
        runtime = app.runtime

        drawerLayout = findViewById(R.id.drawer_layout)
        toolbar = findViewById(R.id.toolbar)

        // Set scrim overlay color
        drawerLayout.setScrimColor(android.graphics.Color.parseColor("#60000000"))

        setupToolbar()
        setupSidebar()

        // Request necessary permissions
        requestNecessaryPermissions()

        // Modern back press handling
        onBackPressedDispatcher.addCallback(this,
            object : androidx.activity.OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (drawerLayout.isDrawerOpen(android.view.Gravity.START)) {
                        drawerLayout.closeDrawer(android.view.Gravity.START)
                    } else {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        )

        // Restore the fragment that was active before recreate (or default to chat)
        if (supportFragmentManager.findFragmentById(R.id.fragment_container) == null) {
            val restored = savedInstanceState?.getString("fragment_tag", "chat") ?: "chat"
            currentFragmentTag = restored
            when (restored) {
                "models" -> navigateTo(ModelsFragment(), "Models")
                "downloads" -> navigateTo(DownloadsFragment(), "Downloads")
                "api" -> navigateTo(ApiFragment(), "API Endpoint")
                else -> navigateTo(ChatFragment(), "PhonoLab")
            }
        }

        refreshSessionList()
    }

    private fun setupToolbar() {
        toolbar.setNavigationOnClickListener {
            drawerLayout.openDrawer(android.view.Gravity.START)
        }
        toolbar.title = "PhonoLab"
    }

    private fun setupSidebar() {
        val rvSessions = findViewById<RecyclerView>(R.id.rv_sessions)
        sessionAdapter = SessionAdapter(
            onSessionClick = { session -> onSessionSelected(session) },
            onSessionLongClick = { session -> onSessionLongPress(session) },
        )
        rvSessions.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = sessionAdapter
        }

        // New Chat button
        findViewById<MaterialButton>(R.id.btn_new_chat).setOnClickListener {
            newChat()
        }

        // Search
        val searchInput = findViewById<android.widget.EditText>(R.id.search_input)
        searchInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) {
                val query = s?.toString() ?: ""
                sessionAdapter.filter(query, sessions)
            }
        })

        // Sidebar nav items
        findViewById<View>(R.id.nav_models).setOnClickListener {
            navigateTo(ModelsFragment(), "Models")
            drawerLayout.closeDrawer(android.view.Gravity.START)
        }
        findViewById<View>(R.id.nav_downloads).setOnClickListener {
            navigateTo(DownloadsFragment(), "Downloads")
            drawerLayout.closeDrawer(android.view.Gravity.START)
        }
        findViewById<View>(R.id.nav_api)?.setOnClickListener {
            navigateTo(ApiFragment(), "API Endpoint")
            drawerLayout.closeDrawer(android.view.Gravity.START)
        }
        findViewById<View>(R.id.nav_logs)?.setOnClickListener {
            val chatFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
            if (chatFragment is ChatFragment) {
                chatFragment.showLogs()
            } else {
                navigateTo(ChatFragment(), "PhonoLab")
            }
            drawerLayout.closeDrawer(android.view.Gravity.START)
        }
        findViewById<View>(R.id.nav_theme_toggle).setOnClickListener {
            ThemeManager.toggleTheme()
            recreate()
        }
    }

    private fun navigateTo(fragment: Fragment, title: String) {
        currentFragmentTag = when (fragment) {
            is ChatFragment -> "chat"
            is ModelsFragment -> "models"
            is DownloadsFragment -> "downloads"
            is ApiFragment -> "api"
            else -> "chat"
        }
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
        toolbar.title = title
    }

    fun refreshSessionList() {
        if (!::sessionManager.isInitialized) return
        sessions = sessionManager.loadAll()
        if (::sessionAdapter.isInitialized) {
            sessionAdapter.setSessions(sessions, activeSessionId)
        }
    }

    override fun onSessionChanged() {
        refreshSessionList()
    }

    private fun newChat() {
        val chatFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
        if (chatFragment is ChatFragment) {
            chatFragment.newChat()
            activeSessionId = chatFragment.getCurrentSession()?.id ?: ""
        } else {
            val newFragment = ChatFragment()
            navigateTo(newFragment, "PhonoLab")
        }
        drawerLayout.closeDrawer(android.view.Gravity.START)
        refreshSessionList()
    }

    private fun onSessionSelected(session: ChatSession) {
        activeSessionId = session.id

        val chatFragment = ChatFragment()
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, chatFragment)
            .commit()
        toolbar.title = "PhonoLab"

        chatFragment.view?.post {
            chatFragment.loadSession(session)
        } ?: run {
            supportFragmentManager.executePendingTransactions()
            chatFragment.loadSession(session)
        }

        drawerLayout.closeDrawer(android.view.Gravity.START)
        refreshSessionList()
    }

    private fun onSessionLongPress(session: ChatSession) {
        val options = arrayOf("Rename", "Delete", "Export")
        android.app.AlertDialog.Builder(this)
            .setTitle(session.title)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> renameSession(session)
                    1 -> deleteSession(session)
                    2 -> exportSession(session)
                }
            }
            .show()
    }

    private fun renameSession(session: ChatSession) {
        val input = android.widget.EditText(this).apply {
            setText(session.title)
            setPadding(48, 32, 48, 32)
        }
        android.app.AlertDialog.Builder(this)
            .setTitle("Rename Session")
            .setView(input)
            .setPositiveButton("Rename") { _, _ ->
                val newTitle = input.text.toString().trim()
                if (newTitle.isNotEmpty()) {
                    sessionManager.rename(session.id, newTitle)
                    refreshSessionList()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteSession(session: ChatSession) {
        android.app.AlertDialog.Builder(this)
            .setTitle("Delete Session")
            .setMessage("Delete \"${session.title}\"?")
            .setPositiveButton("Delete") { _, _ ->
                sessionManager.delete(session.id)
                if (activeSessionId == session.id) {
                    activeSessionId = ""
                    val chatFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                    if (chatFragment is ChatFragment) chatFragment.newChat()
                }
                refreshSessionList()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun exportSession(session: ChatSession) {
        val md = sessionManager.exportMarkdown(session)
        val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(android.content.Intent.EXTRA_TEXT, md)
            putExtra(android.content.Intent.EXTRA_SUBJECT, session.title)
        }
        startActivity(android.content.Intent.createChooser(intent, "Export Chat"))
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("fragment_tag", currentFragmentTag)
    }

    override fun onStop() {
        super.onStop()
        if (!isSelectingImage && isFinishing) {
            try {
                if (::runtime.isInitialized) {
                    runtime.unload()
                }
            } catch (_: Exception) { }
        }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (!isSelectingImage && level >= android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE) {
            try {
                if (::runtime.isInitialized) {
                    runtime.unload()
                }
            } catch (_: Exception) { }
        }
    }

    override fun onDestroy() {
        if (isFinishing) {
            try {
                if (::runtime.isInitialized) {
                    runtime.unload()
                    runtime.killAllLlamaProcesses()
                }
            } catch (_: Exception) { }
        }
        super.onDestroy()
    }

    private fun requestNecessaryPermissions() {
        val needed = mutableListOf<String>()

        when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE -> {
                if (!hasPermission(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)) {
                    needed.add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
                }
                if (!hasPermission(Manifest.permission.READ_MEDIA_IMAGES)) {
                    needed.add(Manifest.permission.READ_MEDIA_IMAGES)
                }
            }
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> {
                if (!hasPermission(Manifest.permission.READ_MEDIA_IMAGES)) {
                    needed.add(Manifest.permission.READ_MEDIA_IMAGES)
                }
            }
            else -> {
                if (!hasPermission(Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    needed.add(Manifest.permission.READ_EXTERNAL_STORAGE)
                }
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.Q) {
                    if (!hasPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                        needed.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    }
                }
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (!hasPermission(Manifest.permission.POST_NOTIFICATIONS)) {
                needed.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        if (needed.isNotEmpty()) {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }
}
