<div align="center">

<img src="../nativelab/icon.png" alt="NativeLab" width="120" height="120" />

# Documentación de NativeLab

</div>

Bienvenido. La documentación está organizada para que cada página cubra un único tema. Elige la que corresponda a lo que deseas hacer y omite el resto.

---

## 🚀 Primeros pasos

| Página                                                         | Cuándo leerla                                                                          |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [installation.md](installation.md)                             | Primera instalación, configuración de llama.cpp y estructura de la carpeta de trabajo. |
| [cli.md](cli.md)                                               | Si deseas utilizar el cliente de terminal (`nativelab --cli`).                         |
| [../nativelab/cli/cli_guide.md](../nativelab/cli/cli_guide.md) | **Guía para principiantes** de la CLI: sencilla y paso a paso.                         |
| [troubleshooting.md](troubleshooting.md)                       | Si algo no funciona y quieres solucionarlo rápidamente.                                |

---

## 🧭 Qué incluye NativeLab

| Página                                     | Tema                                                                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| [features.md](features.md)                 | Catálogo completo de funciones. Las notas de la versión más reciente se encuentran en `changelog.txt`.         |
| [pipeline-builder.md](pipeline-builder.md) | Constructor visual de pipelines, AI Builder, ejemplos predefinidos, esquema JSON y núcleo nativo del pipeline. |
| [architecture.md](architecture.md)         | Diseño por capas, backend centralizado, capa del motor y estructura del proyecto.                              |
| [labs.md](labs.md)                         | La capa de experimentación Labs y cómo añadir nuevas funciones.                                                |
| [integrations.md](integrations.md)         | Rutas de endpoints externos, puente HTTP local y conectores para bots de Discord y WhatsApp.                   |
| [models.md](models.md)                     | Registro de modelos, detección de familias, cuantizaciones y modelos API.                                      |
| [workflows.md](workflows.md)               | Pipelines, referencias, resumen de documentos, MCP y descarga de modelos/runtime.                              |
| [ui.md](ui.md)                             | Componentes de la interfaz gráfica, temas, atajos y persistencia.                                              |

---

## 📱 PhonoLab - Cliente para Android

PhonoLab es el cliente oficial de NativeLab para Android. Mantiene la misma filosofía **local-first** y ejecuta llama.cpp directamente en el dispositivo con funciones avanzadas específicas para móviles.

| Página                                                | Tema                                                                                                            |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [PhonoLab README](../PhonoLab/docs/README.md)         | Índice de la documentación de la aplicación Android.                                                            |
| [ANDROID_APP.md](../PhonoLab/docs/ANDROID_APP.md)     | Arquitectura Android, manejo de errores, patrón Singleton y protección del ciclo de vida.                       |
| [FILE_INDEX.md](../PhonoLab/docs/FILE_INDEX.md)       | Todos los archivos del proyecto Android: propósito, clases y constantes.                                        |
| [CONSTANTS.md](../PhonoLab/docs/CONSTANTS.md)         | Todas las constantes, límites, colores y URL de Android.                                                        |
| [CONTRIBUTING.md](../PhonoLab/docs/CONTRIBUTING.md)   | Qué modificar para realizar cambios específicos en Android.                                                     |
| [api_endpoints.md](../PhonoLab/docs/api_endpoints.md) | Referencia completa de la API de PhonoLab: todos los endpoints, ejemplos con curl, visión, RAG y configuración. |
| [cross-platform.md](cross-platform.md)                | Cómo trabajan juntos NativeLab y PhonoLab en distintas plataformas.                                             |

## 🌐 Integración multiplataforma

NativeLab y PhonoLab están diseñados para funcionar de forma integrada entre escritorio y dispositivos móviles:

* **Descubrimiento de dispositivos en la LAN:** NativeLab puede detectar y utilizar dispositivos PhonoLab como servidores remotos de IA.
* **Ecosistema compartido de modelos:** mismo formato GGUF y los mismos niveles de cuantización.
* **Estándares API unificados:** endpoints compatibles con OpenAI y Anthropic en ambas plataformas.
* **Flujos de trabajo multiplataforma:** crea pipelines que abarquen tanto equipos de escritorio como dispositivos móviles.

Más información en la documentación de [cross-platform.md](cross-platform.md).

---

## 🗂️ Enlaces rápidos por tarea

**"Quiero chatear con un modelo local desde mi terminal."**
→ [cli.md](cli.md) y la [guía para principiantes de la CLI](../nativelab/cli/cli_guide.md).

**"Quiero construir un pipeline con múltiples pasos, ramas y bucles."**
→ [pipeline-builder.md](pipeline-builder.md).

**"Quiero que la aplicación genere un pipeline a partir de una descripción."**
→ [pipeline-builder.md#ai-builder-tab](pipeline-builder.md#ai-builder-tab).

**"Quiero enviar PDFs largos a un modelo."**
→ [workflows.md#summarization-pipeline](workflows.md#summarization-pipeline).

**"Quiero conectar OpenAI, Anthropic o una instancia local de Ollama."**
→ [models.md#local-and-api-backend-support](models.md#local-and-api-backend-support).

**"Quiero desarrollar una nueva función experimental."**
→ [labs.md](labs.md).

**"Quiero conectar NativeLab con Discord, WhatsApp, webhooks o un script local."**
→ [integrations.md](integrations.md).

**"Quiero entender cómo está organizado el código del proyecto."**
→ [architecture.md](architecture.md).

**"Quiero ejecutar NativeLab en mi teléfono Android."**
→ [Documentación de PhonoLab](../PhonoLab/docs/README.md) y la [página web de PhonoLab](../web_page/phonolab.html).

**"Quiero utilizar mi teléfono como servidor de IA en la LAN desde NativeLab."**
→ [API de PhonoLab](../PhonoLab/docs/api_endpoints.md): escanea y registra dispositivos desde **Dev → Devices**. La autenticación inteligente gestiona automáticamente las claves.

**"Quiero crear flujos de trabajo multiplataforma utilizando tanto el escritorio como el móvil."**
→ [cross-platform.md](cross-platform.md): cómo trabajan juntos NativeLab y PhonoLab entre distintas plataformas.

---

## 📜 Otros documentos

* [LICENSE](../LICENSE) — Licencia AGPL v3.
* [CONTRIBUTING.md](../CONTRIBUTING.md) — Cómo enviar una Pull Request.
* [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Normas de la comunidad.
* [SECURITY.md](../SECURITY.md) — Cómo informar vulnerabilidades.
