import unittest
import tempfile
import shutil
import cv2
import numpy as np
from unittest.mock import patch
from photo_separator_gui import PhotoSeparatorGUI


class TestPhotoSeparatorGUI(unittest.TestCase):
    """
    Testes para comparações de fotos e funcionalidades do GUI.
    by @gustavosett #rev 02/04/2023
    """

    def setUp(self):
        self.app = PhotoSeparatorGUI()
        self.temp_input_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_input_dir)
        shutil.rmtree(self.temp_output_dir)

    def test_compare_faces_different(self):
        # Criando duas imagens diferentes artificialmente
        face1 = np.random.rand(150, 150, 3) * 255
        face2 = np.random.rand(150, 150, 3) * 255
        self.assertFalse(self.app.compare_faces(face1, face2))

    def test_compare_faces_error_handling(self):
        # Testando o manejo de erro ao comparar faces
        self.app.compare_faces(None, None)

    def test_mainloop(self):
        with patch('tkinter.Tk.mainloop'):
            self.app.mainloop()

    def test_select_input_dir(self):
        # Testando a seleção de diretório de entrada e alteração da variável
        with patch('tkinter.filedialog.askdirectory', return_value=self.temp_input_dir):
            self.app.select_input_dir()
            self.assertEqual(self.app.input_dir_var.get(), self.temp_input_dir)

    def test_select_output_dir(self):
        # Testando a seleção de diretório de saída e alteração da variável
        with patch('tkinter.filedialog.askdirectory', return_value=self.temp_output_dir):
            self.app.select_output_dir()
            self.assertEqual(self.app.output_dir_var.get(), self.temp_output_dir)

if __name__ == '__main__':
    unittest.main()
