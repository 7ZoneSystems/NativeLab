try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from PyPDF2 import PdfReader; HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import hashlib, pickle, struct
    HAS_HASH = True
except ImportError:
    HAS_HASH = False