# Integración Multiplataforma

NativeLab y PhonoLab están diseñados para trabajar juntos de forma fluida entre plataformas de escritorio y móviles, creando un ecosistema de IA **local-first** unificado.

## Descripción general de la arquitectura

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          NativeLab Desktop                              │
│  • GUI basada en PyQt6 con constructor de pipelines, Labs y funciones   │
│    avanzadas                                                            │
│  • Cliente CLI para usuarios de terminal                                │
│  • Inferencia local mediante llama.cpp (servidor o CLI)                 │
│  • Integraciones con APIs (OpenAI, Anthropic, Ollama)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PhonoLab Android                               │
│  • Inferencia local con llama.cpp mediante binarios integrados          │
│  • Compatibilidad con modelos de visión (LLaVA, Qwen-VL, etc.)          │
│  • Procesamiento RAG para PDF, documentos y archivos de texto           │
│  • Servidor API LAN para comunicación entre escritorio y móvil          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Puntos clave de integración

### 1. Descubrimiento de dispositivos LAN e integración API

NativeLab puede descubrir automáticamente dispositivos PhonoLab en la misma red local y utilizarlos como servidores de IA remotos:

* **Descubrimiento de dispositivos:** La pestaña **Devices** de NativeLab escanea la red LAN en busca de servidores API de PhonoLab.
* **Autenticación inteligente:** Intercambio y gestión automática de claves.
* **Integración transparente:** PhonoLab aparece como un modelo API normal dentro del selector de modelos de NativeLab.
* **Flujos de trabajo multiplataforma:** Utiliza NativeLab en escritorio para orquestar pipelines que incluyan inferencia realizada por PhonoLab en dispositivos móviles.

### 2. Ecosistema de modelos compartido

Ambas plataformas utilizan el mismo formato de modelos GGUF y admiten los mismos niveles de cuantización.

* **Compatibilidad de modelos:** Los modelos descargados en escritorio pueden transferirse a Android y viceversa.
* **Catálogo compartido:** El mismo catálogo de modelos (SmolLM2, Qwen2.5, Llama 3.2, etc.) está disponible en ambas plataformas.
* **Configuración consistente:** Los mismos parámetros (temperatura, `top_p`, tamaño del contexto) funcionan de forma idéntica.

### 3. Estándares API unificados

PhonoLab implementa los mismos estándares API de OpenAI y Anthropic que las integraciones API de NativeLab.

* **Endpoints compatibles con OpenAI:** `/v1/chat/completions`, `/v1/models`, `/v1/health`
* **Endpoints compatibles con Anthropic:** `/v1/messages`, `/v1/health`
* **Compatibilidad con streaming:** Eventos enviados por el servidor (SSE) para transmisión de tokens en tiempo real.
* **Compatibilidad con visión:** El mismo formato de URL de imágenes para solicitudes multimodales.

### 4. Documentación compartida y patrones de desarrollo

* **Manejo consistente de errores:** Los mismos patrones de manejo de errores multinivel (fatales y no fatales).
* **Configuración unificada:** Estructuras de configuración y valores predeterminados similares.
* **Buenas prácticas compartidas:** Renderizado matemático, endurecimiento de seguridad y optimizaciones de rendimiento.
* **Flujo de desarrollo común:** Los mismos estándares para pruebas, CI/CD y documentación.

## Primeros pasos con flujos de trabajo multiplataforma

### Usar PhonoLab como servidor remoto desde NativeLab

1. Instala PhonoLab en tu dispositivo Android.
2. Activa el servidor API en la pestaña **API Endpoint** de PhonoLab.
3. En NativeLab Desktop, abre **Dev → Devices**.
4. Haz clic en **Scan Network** para descubrir tu dispositivo PhonoLab.
5. Selecciona **Register as Model** para añadirlo al registro de modelos.
6. Elígelo desde el selector de modelos y úsalo como cualquier otro modelo.

### Transferencia de modelos entre plataformas

* **De escritorio a móvil:** Descarga modelos con el gestor de modelos de NativeLab y transfiere los archivos `.gguf` a tu dispositivo Android.
* **De móvil a escritorio:** Descarga modelos desde PhonoLab y transfiérelos mediante USB o almacenamiento en la nube.

### Desarrollo multiplataforma

Al desarrollar nuevas funciones:

* **Primero escritorio:** Implementa la lógica principal en Python para NativeLab.
* **Adaptación móvil:** Porta la implementación a Kotlin/Java para PhonoLab utilizando enlaces JNI.
* **Bibliotecas compartidas:** Extrae la lógica común a bibliotecas compartidas siempre que sea posible.
* **Pruebas consistentes:** Verifica las funciones en ambas plataformas antes del lanzamiento.

## Capacidades específicas de cada plataforma

| Función                 | NativeLab Desktop                                                        | PhonoLab Android                                                                   |
| ----------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Tamaño de modelos**   | Admite modelos grandes (hasta 13B o más en equipos de alto rendimiento). | Optimizado para dispositivos móviles (hasta 3B en la mayoría de los dispositivos). |
| **Modelos de visión**   | Compatibles mediante integraciones API.                                  | Compatibilidad nativa con modelos de visión ejecutados localmente.                 |
| **Procesamiento RAG**   | Procesamiento de PDF y documentos mediante bibliotecas Python.           | Procesamiento nativo de PDF, texto y DOCX en el dispositivo.                       |
| **Almacenamiento**      | Acceso directo al sistema de archivos.                                   | SAF (Storage Access Framework) con gestión de permisos.                            |
| **Interfaz de usuario** | Constructor avanzado de pipelines, Labs y herramientas de visualización. | Interfaz de chat optimizada para dispositivos móviles.                             |
| **Rendimiento**         | Mayor velocidad de procesamiento y ventanas de contexto más grandes.     | Optimizado para el consumo de batería y memoria.                                   |

## Solución de problemas multiplataforma

### Problemas comunes y soluciones

* **El dispositivo no aparece:** Asegúrate de que ambos dispositivos estén conectados a la misma red Wi-Fi y que el servidor API de PhonoLab esté habilitado.
* **Tiempos de espera agotados:** Comprueba la configuración del firewall y verifica que PhonoLab tenga permisos de acceso a la red.
* **Errores al cargar modelos:** Comprueba la compatibilidad del modelo (arquitectura y nivel de cuantización).
* **Errores de autenticación:** Restablece las claves API tanto en PhonoLab como en NativeLab.
* **Problemas con el streaming:** Verifica la estabilidad de la red e intenta reducir el tamaño del contexto.

### Comandos de diagnóstico

```bash
# Probar la conectividad con el servidor API de PhonoLab
curl http://[PHONO_LAB_IP]:8080/health

# Listar los modelos disponibles en PhonoLab
curl http://[PHONO_LAB_IP]:8080/v1/models

# Probar una solicitud simple de chat
curl -X POST http://[PHONO_LAB_IP]:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phonolab-active", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

## Planes futuros de integración

* **Sesiones sincronizadas:** Compartir conversaciones entre escritorio y móvil.
* **Registro de modelos unificado:** Catálogo de modelos sincronizado mediante la nube.
* **Pipelines multiplataforma:** Crear pipelines que se ejecuten entre dispositivos de escritorio y móviles.
* **Labs compartidos:** Funciones experimentales disponibles en ambas plataformas.
* **Sistema de actualización unificado:** Un único mecanismo de actualización para NativeLab y PhonoLab.

---

Esta integración multiplataforma convierte a NativeLab y PhonoLab en un ecosistema completo para el desarrollo y uso de IA local, desde potentes flujos de trabajo en escritorio hasta una inferencia cómoda y eficiente en dispositivos móviles.
