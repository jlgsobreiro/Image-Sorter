"""Regressões de renderização de miniaturas em janelas Tk independentes."""
import gc
import os
import tempfile
import tkinter as tk
import unittest

from PIL import Image

from db import init_db, insert_images_batch
from duplicate_finder import DuplicateReviewGUI


class TestDuplicateReviewGUI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db_path = os.path.join(directory.name, "duplicates.db")
        init_db(self.db_path)
        records = []
        for index, color in enumerate(["red", "red", "blue", "blue"]):
            path = os.path.join(directory.name, f"photo_{index}.png")
            with Image.new("RGB", (80, 60), color) as image:
                image.save(path)
            records.append({"file_path": path, "file_name": os.path.basename(path),
                            "file_size": os.path.getsize(path)})
        insert_images_batch(self.db_path, records)

    def _create_reviewer(self):
        gui = DuplicateReviewGUI.__new__(DuplicateReviewGUI)
        # Garante limpeza mesmo se a renderização falhar durante __init__.
        def cleanup():
            if hasattr(gui, "root"):
                try:
                    gui.root.destroy()
                except tk.TclError:
                    pass  # A janela pode ter sido fechada pelo próprio teste.
        self.addCleanup(cleanup)
        gui.__init__(db_path=self.db_path)
        gui.root.withdraw()
        return gui

    def _assert_thumbnails(self, gui):
        gc.collect()
        self.assertEqual(len(gui.tk_images), 2)
        image_names = gui.root.tk.call("image", "names")
        for image in gui.tk_images:
            self.assertIs(image.tk, gui.root.tk)
            self.assertIn(str(image), image_names)
            self.assertGreater(int(gui.root.tk.call("image", "width", str(image))), 0)

    def test_standalone_review_and_navigation(self):
        gui = self._create_reviewer()
        self.assertEqual(len(gui.groups), 2)
        self._assert_thumbnails(gui)
        gui._next_group()
        self.assertEqual(gui.current_group_idx, 1)
        self._assert_thumbnails(gui)
        gui._prev_group()
        self.assertEqual(gui.current_group_idx, 0)
        self._assert_thumbnails(gui)

    def test_review_with_another_tk_window_and_reopening(self):
        dashboard = tk.Tk()
        dashboard.withdraw()
        self.addCleanup(dashboard.destroy)
        for _ in range(2):
            gui = self._create_reviewer()
            self.assertIsNot(gui.root.tk, dashboard.tk)
            self._assert_thumbnails(gui)
            gui._next_group()
            self._assert_thumbnails(gui)
            gui._on_close()
            self.assertTrue(dashboard.winfo_exists())


if __name__ == "__main__":
    unittest.main()