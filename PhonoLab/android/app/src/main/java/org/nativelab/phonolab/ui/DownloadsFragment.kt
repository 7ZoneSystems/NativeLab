package org.nativelab.phonolab.ui

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.*
import org.nativelab.phonolab.adapter.CatalogAdapter
import org.nativelab.phonolab.data.ModelManager
import org.nativelab.phonolab.util.UiHelpers
import java.util.concurrent.Executors

class DownloadsFragment : Fragment() {

    private lateinit var store: PhonoLabStore
    private lateinit var runtime: LlamaRuntime
    private lateinit var modelManager: ModelManager

    private lateinit var runtimeStatus: TextView
    private lateinit var runtimeProgress: ProgressBar

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    private fun runOnUi(block: () -> Unit) {
        main.post { if (isAdded) block() }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_downloads, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val app = requireActivity().application as PhonoLabApp
        store = app.store
        runtime = app.runtime
        modelManager = app.modelManager

        runtimeStatus = view.findViewById(R.id.runtime_status)
        runtimeProgress = view.findViewById(R.id.runtime_progress)

        view.findViewById<MaterialButton>(R.id.btn_pull_runtime).setOnClickListener { installRuntime() }

        val rvCatalog = view.findViewById<RecyclerView>(R.id.rv_catalog)
        rvCatalog.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = CatalogAdapter { candidate -> downloadCatalogModel(candidate) }
        }

        val rvDownloads = view.findViewById<RecyclerView>(R.id.rv_downloads)
        rvDownloads.layoutManager = LinearLayoutManager(context)

        refreshRuntimeStatus()
    }

    override fun onDestroyView() {
        worker.shutdown()
        super.onDestroyView()
    }

    private fun refreshRuntimeStatus() {
        val rt = runtime.runtimeStatus()
        val arch = runtime.cppManager.detectArch()
        val archInfo = "Device: ${arch.primary} | ABIs: ${arch.allAbis.joinToString()}"
        if (rt.ready) {
            val verLine = rt.version.lines().firstOrNull()?.take(80) ?: ""
            runtimeStatus.text = "✅ ${rt.message}\n${rt.serverPath}\n$verLine\n$archInfo"
            runtimeStatus.setTextColor(resources.getColor(R.color.ph_ok, null))
        } else {
            val suffix = runtime.cppManager.archDownloadSuffix()
            val statusMsg = if (suffix == null) {
                "⚠️ No llama.cpp build for ${arch.primary}\nllama.cpp only supports arm64-v8a\n$archInfo"
            } else {
                "⚠️ ${rt.message}\nTap 'Install Runtime' to download.\n$archInfo"
            }
            runtimeStatus.text = statusMsg
            runtimeStatus.setTextColor(resources.getColor(R.color.ph_warn, null))
        }
    }

    private fun installRuntime() {
        worker.execute {
            try {
                runOnUi {
                    runtimeProgress.visibility = View.VISIBLE
                    runtimeProgress.progress = 0
                    runtimeStatus.text = "Downloading llama-server…"
                    runtimeStatus.setTextColor(resources.getColor(R.color.ph_txt2, null))
                }

                val server = runtime.autoInstallBinary { done, total, label ->
                    runOnUi {
                        val pct = UiHelpers.calcPercent(done, total)
                        runtimeProgress.progress = pct
                        runtimeStatus.text = "Installing $label: $pct%"
                    }
                }

                runOnUi {
                    runtimeProgress.visibility = View.GONE
                    if (server != null) {
                        refreshRuntimeStatus()
                        Toast.makeText(context, "llama-server installed!", Toast.LENGTH_SHORT).show()
                    } else {
                        runtimeStatus.text = "❌ Install returned no binary."
                        runtimeStatus.setTextColor(resources.getColor(R.color.ph_err, null))
                    }
                }
            } catch (e: Exception) {
                runOnUi {
                    runtimeProgress.visibility = View.GONE
                    runtimeStatus.text = "❌ ${e.javaClass.simpleName}: ${e.message}"
                    runtimeStatus.setTextColor(resources.getColor(R.color.ph_err, null))
                }
            }
        }
    }

    private fun downloadCatalogModel(candidate: ModelCandidate) {
        worker.execute {
            try {
                runOnUi {
                    runtimeProgress.visibility = View.VISIBLE
                    runtimeProgress.progress = 0
                    runtimeStatus.text = "Downloading ${candidate.label}…"
                    runtimeStatus.setTextColor(resources.getColor(R.color.ph_txt2, null))
                }

                val downloader = SafeDownloader(store)
                val model = downloader.downloadModel(candidate) { done, total, label ->
                    runOnUi {
                        val pct = UiHelpers.calcPercent(done, total)
                        runtimeProgress.progress = pct
                        runtimeStatus.text = "$label: $pct%"
                    }
                }

                modelManager.add(model, repo = candidate.repo)

                runOnUi {
                    runtimeProgress.visibility = View.GONE
                    val sizeMb = model.length() / (1024 * 1024)
                    runtimeStatus.text = "✅ ${model.name} ($sizeMb MB) — ready to load"
                    runtimeStatus.setTextColor(resources.getColor(R.color.ph_ok, null))
                    Toast.makeText(context, "Model ready! Go to Chat and tap Load.", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUi {
                    runtimeProgress.visibility = View.GONE
                    runtimeStatus.text = "Download failed: ${e.message}"
                    runtimeStatus.setTextColor(resources.getColor(R.color.ph_err, null))
                }
            }
        }
    }
}
