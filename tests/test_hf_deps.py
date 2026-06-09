import sys
import unittest

from nativelab.Server.hf_deps import (
    HF_TRANSFORMERS_DEP_IMPORTS,
    HF_TRANSFORMERS_DEP_PACKAGES,
    build_hf_transformers_install_command,
    hf_transformers_dependency_report,
)


class HfTransformersDepsTests(unittest.TestCase):
    def test_install_command_uses_current_python_and_explicit_packages(self):
        cmd = build_hf_transformers_install_command()
        self.assertEqual(cmd[:5], [sys.executable, "-m", "pip", "install", "-U"])
        self.assertIn("transformers>=5.0.0", cmd)
        self.assertIn("torch>=2.2.0", cmd)
        self.assertIn("accelerate>=0.30.0", cmd)

    def test_dependency_registry_has_runtime_imports(self):
        packages = "\n".join(HF_TRANSFORMERS_DEP_PACKAGES)
        imports = dict(HF_TRANSFORMERS_DEP_IMPORTS)
        self.assertIn("safetensors>=0.4.0", packages)
        self.assertEqual(imports["pillow"], "PIL")
        self.assertEqual(imports["transformers"], "transformers")

    def test_dependency_report_shape_is_stable(self):
        report = hf_transformers_dependency_report()
        self.assertIn("ok", report)
        self.assertIn("installed", report)
        self.assertIn("missing", report)
        self.assertIn("packages", report)
        self.assertIsInstance(report["missing"], list)


if __name__ == "__main__":
    unittest.main()
