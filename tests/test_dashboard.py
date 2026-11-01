import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_dashboard_load_filter_and_engine_selection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Predictive Maintenance Lab")
        total = len(app.dataframe[0].value)
        self.assertGreater(total, 0)
        app.checkbox[0].check().run()
        self.assertEqual(len(app.exception), 0)
        filtered = app.dataframe[0].value
        self.assertTrue((filtered.alert == 1).all())
        self.assertLessEqual(len(filtered), total)
        app.selectbox[0].select(2).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Moteur **2**" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
