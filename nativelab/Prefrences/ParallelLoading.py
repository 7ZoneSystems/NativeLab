from nativelab.imports.import_global import dataclass, List, json, field
from nativelab.GlobalConfig.config_global import PARALLEL_PREFS_FILE
@dataclass
class ParallelPrefs:
    enabled:           bool = False
    auto_load_roles:   List[str] = field(default_factory=list)
    pipeline_mode:     bool = False   # reasoning → coding chain
    warned:            bool = False

    def save(self):
        PARALLEL_PREFS_FILE.write_text(json.dumps({
            "enabled": self.enabled,
            "auto_load_roles": self.auto_load_roles,
            "pipeline_mode": self.pipeline_mode,
            "warned": self.warned,
        }, indent=2))

    @classmethod
    def load(cls) -> "ParallelPrefs":
        if PARALLEL_PREFS_FILE.exists():
            try:
                d = json.loads(PARALLEL_PREFS_FILE.read_text())
                return cls(**{k: v for k, v in d.items()
                               if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()


PARALLEL_PREFS = ParallelPrefs.load()
