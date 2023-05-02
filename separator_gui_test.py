import unittest
import tempfile
import shutil
import cv2
from pathlib import Path
from unittest.mock import patch
from photo_separator_gui import PhotoSeparatorGUI


class TestPhotoSeparatorGUI(unittest.TestCase):
    """
    Testes comparações de fotos.
    by @gustavosett #rev 02/04/2023
    """

    def setUp(self):
        self.app = PhotoSeparatorGUI()
        self.temp_input_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_input_dir)
        shutil.rmtree(self.temp_output_dir)

    def test_compare_faces_same(self):
        face1 = cv2.imread("face1.jpg")
        face2 = cv2.imread("face1.jpg")
        self.assertTrue(self.app.compare_faces(face1, face2))

    def test_compare_faces_different(self):
        face1 = cv2.imread("face1.jpg")
        face2 = cv2.imread("face2.jpg")
        self.assertFalse(self.app.compare_faces(face1, face2))

    def test_mainloop(self):
        with patch('tkinter.Tk.mainloop'):
            self.app.mainloop()


if __name__ == '__main__':
    unittest.main()
