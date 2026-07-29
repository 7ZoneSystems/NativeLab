# Instalación

NativeLab funciona en **Linux, macOS y Windows**. La instalación y los comandos son los mismos en todas las plataformas.

---

## 1. Instalar el paquete de Python

```bash id="uw2wib"
pip install nativelab
```

La instalación predeterminada corresponde a la versión con interfaz gráfica (GUI). Incluye `PyQt6` para la aplicación de escritorio y `psutil` / `pypdf` para la supervisión en tiempo real de la memoria RAM y el soporte para archivos PDF.

La CLI se instala junto con la GUI.

Para instalaciones desde el código fuente o para desarrollo, el proyecto también define los grupos de dependencias opcionales `cli` y `labs`. Sin embargo, la instalación estándar desde PyPI mostrada arriba sigue siendo la opción recomendada para la mayoría de los usuarios.

> **Versión de Python:** Se requiere Python 3.10 o superior.

Verifica la instalación:

```bash id="cjlwmr"
nativelab --help
nativelab --cli --help
```

---

## 2. Elegir una carpeta de trabajo

NativeLab guarda todos sus datos en el **directorio de trabajo actual**; no crea archivos de configuración ocultos en tu directorio personal.

Crea una carpeta que utilizarás como directorio de trabajo predeterminado:

```bash id="8g3gkn"
mkdir ~/nativelab
cd ~/nativelab
```

Todo lo que genere NativeLab (modelos, sesiones, configuraciones y trabajos pausados) se almacenará aquí.

Mover esta carpeta sirve como copia de seguridad completa; eliminarla permite comenzar desde cero.

---

## 3. Instalar los binarios de llama.cpp

Necesitas disponer de `llama-server` y `llama-cli` en algún lugar del disco.

Existen tres formas de obtenerlos.

### Opción A — Utilizar el instalador integrado de la GUI (recomendado)

```bash id="2bjlwm"
nativelab
```

En la aplicación de escritorio, abre la pestaña **⬇️ Download**.

Selecciona la versión más reciente de llama.cpp para tu plataforma y pulsa **Install**.

Los binarios se instalarán automáticamente en:

```text id="3qut7i"
./llama/bin/
```

Los nuevos usuarios también pueden utilizar el asistente de configuración automática que aparece durante la primera ejecución.

Este asistente:

* Analiza la memoria RAM, la CPU y los backends GPU disponibles.
* Ofrece una configuración basada en llama.cpp GGUF o en Hugging Face Transformers.
* Puede reanudarse después de una pausa, cancelación, cierre inesperado o reinicio.

---

### Opción B — Descargar manualmente el paquete oficial

Descarga el archivo correspondiente desde:

https://github.com/ggml-org/llama.cpp/releases

Extrae el contenido dentro de:

```text id="g41hxe"
llama/bin/
├── llama-cli
├── llama-server
└── (bibliotecas auxiliares)
```

En Linux y macOS, concede permisos de ejecución:

```bash id="a0cmif"
chmod -R +x llama/bin/
```

---

### Opción C — Compilar desde el código fuente

```bash id="6lr8zw"
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
mkdir -p ../nativelab/llama/bin
cp build/bin/llama-cli build/bin/llama-server ../nativelab/llama/bin/
```

NativeLab busca los binarios siguiendo este orden:

1. `cli_path` y `server_path` definidos en `server_config.json` (configurables desde la pestaña **Server** de la GUI).
2. Binarios integrados en `llama-bin/` (solo en versiones compiladas).
3. Ruta de desarrollo `./llama/bin/`.

---

## 4. Instalar o descargar un modelo

Necesitas al menos un modelo.

Los modelos **GGUF** son la forma más sencilla de trabajar localmente, aunque NativeLab también puede registrar modelos activos de Ollama y snapshots opcionales de Hugging Face Transformers.

### Desde la GUI

* Abre la pestaña **⬇️ Download**.
* Escribe un repositorio de Hugging Face (por ejemplo, `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`) y pulsa **Search**.
* Descarga una cuantización adecuada para la cantidad de memoria RAM disponible.

La misma pestaña también permite:

* Descargar snapshots completos de Hugging Face Transformers dentro de `localllm/hf_transformers/` y registrarlos como `hf:<local-folder>`.
* Importar modelos desde un daemon de Ollama ya en ejecución y registrarlos como `ollama:<model>`.

Las bibliotecas necesarias para Hugging Face Transformers son opcionales.

En el panel **HF Transformers snapshot**, utiliza **Install Libraries** para instalar o comprobar automáticamente:

* Transformers
* Torch
* safetensors
* Accelerate
* SentencePiece
* Pillow

Los usuarios avanzados pueden ejecutar manualmente el comando `pip` mostrado por dicho panel.

### Desde la CLI

```bash id="0rcm4x"
nativelab --cli setup
```

El asistente ofrece tres modelos iniciales recomendados o permite introducir cualquier repositorio de Hugging Face y seleccionar la cuantización deseada.

### Desde tus propios archivos

Coloca cualquier archivo `.gguf` dentro de:

```text id="pvr1cl"
./localllm/
```

o utiliza el botón **Browse GGUF…** disponible en la pestaña **Models** de la interfaz gráfica.

> **Guía aproximada de memoria:** 7B Q4 ≈ 4.5 GB de RAM · 13B Q5 ≈ 9.5 GB · 70B Q4 ≈ 38 GB. La descarga de trabajo hacia la GPU (**Settings → Server**) reduce el uso de RAM trasladando capas a la memoria de vídeo (VRAM).

---

## 5. Ejecutar NativeLab

```bash id="jcy0an"
nativelab            # GUI
nativelab --cli      # CLI
```

¡Listo!

A partir de aquí:

* ¿Eres nuevo usando terminales? Consulta la **Guía para principiantes de la CLI**: [../nativelab/cli/cli_guide.md](../nativelab/cli/cli_guide.md)
* ¿Quieres conocer todas las funciones? Consulta **features.md**.
* ¿Tienes algún problema? Revisa **troubleshooting.md**.
