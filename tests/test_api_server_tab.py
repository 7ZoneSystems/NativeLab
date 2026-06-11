import os
import unittest

os.environ.setdefault("NATIVELAB_NO_GUI", "1")


class ApiServerTabTests(unittest.TestCase):
    def test_api_server_tab_builds_with_eye_key_buttons_in_headless_mode(self):
        from nativelab.api_server.tab import ApiServerTab

        tab = ApiServerTab(None)

        self.assertTrue(hasattr(tab, "btn_show_local_key"))
        self.assertTrue(hasattr(tab, "btn_show_lan_key"))
        self.assertNotIn("show_keys_toggle", tab.__dict__)
        tab.shutdown()


if __name__ == "__main__":
    unittest.main()
