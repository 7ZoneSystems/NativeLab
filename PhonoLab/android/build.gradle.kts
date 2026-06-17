plugins {
    id("com.android.application") version "8.13.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}

tasks.register("syncLlamaCpp") {
    group = "phonolab"
    description = "Clone or update llama.cpp source for Android runtime development."

    doLast {
        val target = rootProject.file("external/llama.cpp")
        if (target.exists()) {
            exec {
                workingDir = target
                commandLine("git", "pull", "--ff-only")
            }
        } else {
            target.parentFile.mkdirs()
            exec {
                commandLine(
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/ggml-org/llama.cpp.git",
                    target.absolutePath,
                )
            }
        }
    }
}
