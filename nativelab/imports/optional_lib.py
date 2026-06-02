try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from pypdf import PdfReader; HAS_PDF = True
except ImportError:
    PdfReader = None
    HAS_PDF = False

try:
    import hashlib, pickle, struct
    HAS_HASH = True
except ImportError:
    HAS_HASH = False
