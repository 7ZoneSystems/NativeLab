plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.nativelab.phonolab"
    compileSdk = 36

    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "org.nativelab.phonolab"
        minSdk = 21
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // Only package ABIs we have binaries for
            // arm64-v8a: all 64-bit devices (~95% of active devices)
            // armeabi-v7a: 32-bit fallback (older/budget devices)
            // Remove armeabi-v7a if no 32-bit binary available
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    packagingOptions {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // AndroidX core
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.fragment:fragment-ktx:1.8.6")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation("androidx.drawerlayout:drawerlayout:1.2.0")
    implementation("androidx.cardview:cardview:1.0.0")

    // Material Design 3
    implementation("com.google.android.material:material:1.13.0")

    // JSON
    implementation("org.json:json:20260522")

    // DocumentFile for SAF
    implementation("androidx.documentfile:documentfile:1.1.0")

    // Splash screen
    implementation("androidx.core:core-splashscreen:1.2.0-alpha02")
}
