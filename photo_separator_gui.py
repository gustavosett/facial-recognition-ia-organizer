import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import dlib
from unidecode import unidecode


class PhotoSeparatorGUI:
    """
    Interface gráfica de seleção de diretorios de fotos
    e aplicar separação com base em reconhecimento facial.
    @gustavosett #rev 30/04/2023
    """


    def __init__(self):
        # Inicializa a janela principal
        self.root = tk.Tk()
        self.root.title('Separador de Fotos por Reconhecimento Facial')

        # Inicializa as variáveis de diretório
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.faces_dir_var = tk.StringVar()

        # Cria e organiza os widgets na janela
        self.create_widgets()
        self.arrange_widgets()


    def select_input_dir(self):
        """Solicita ao usuário que selecione o diretório de entrada das fotos."""
        input_dir = filedialog.askdirectory()
        if input_dir:
            self.input_dir_var.set(input_dir)
            print('Diretório de entrada selecionado:', input_dir)


    def select_output_dir(self):
        """Solicita ao usuário que selecione o diretório de saída das fotos."""
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_var.set(output_dir)
            print('Diretório de saída selecionado:', output_dir)


    def select_faces_dir(self):
        """Solicita ao usuário que selecione o diretório de treinamento das faces."""
        faces_dir = filedialog.askdirectory()
        if faces_dir:
            self.faces_dir_var.set(faces_dir)
            print('Diretório de treinamento das faces selecionado:', faces_dir)


    def run(self):
        """Executa a separação das fotos."""
        print('Executou!')
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        faces_dir = self.faces_dir_var.get()

        # Verifica se os diretórios foram selecionados corretamente
        if not self.check_directories(input_dir, output_dir, faces_dir):
            return
        
        # Separe as fotos com base no reconhecimento facial
        self.separate_photos(input_dir, output_dir, faces_dir)
        print('Executou corretamente!')


    def print_error(self, title, message):
        """Imprime uma mensagem de erro."""
        messagebox.showerror(title, message)
        print(f'{title}: {message}')


    def check_directories(self, input_dir, output_dir, faces_dir):
        """Verifica se os diretórios foram selecionados, se são distintos e se são válidos."""
        error_messages = [
            ('Por favor, selecione um diretório de entrada.', input_dir),
            ('Por favor, selecione um diretório de saída.', output_dir),
            ('Por favor, selecione o diretório de treinamento.', faces_dir)
        ]

        for error_message, directory in error_messages:
            if not directory:
                self.print_error('Erro', error_message)
                return False

        if len(set([input_dir, output_dir, faces_dir])) != 3:
            self.print_error('Erro', 'Os diretórios selecionados não podem ser iguais. Por favor, escolha diretórios diferentes.')
            return False

        for directory in [input_dir, output_dir, faces_dir]:
            if not os.path.isdir(directory):
                self.print_error('Erro', f'O diretório "{directory}" não é válido. Por favor, selecione um diretório válido.')
                return False

        return True
    

    def separate_photos(self, input_dir, output_dir, faces_dir):
        """Função principal para gerenciar a separação de fotos"""

        # Carregue o detector de faces, o shape predictor e o modelo de reconhecimento facial
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(faces_dir, "shape_predictor_68_face_landmarks.dat"))
        self.facerec = dlib.face_recognition_model_v1(os.path.join(faces_dir, "dlib_face_recognition_resnet_model_v1.dat"))


        # Itere sobre todas as imagens no diretório de entrada
        for filename in os.listdir(input_dir):
            decoded_filename = unidecode(filename)
            src = os.path.join(input_dir, filename)
            dst = os.path.join(input_dir, decoded_filename)
            os.rename(src, dst)
            if not os.path.isfile(src):
                continue

            # Carregue a imagem usando o OpenCV
            image = cv2.imread(src)
            if image is None:
                continue

            # Detecte faces na imagem
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Separe e agrupe cada face encontrada na imagem
            for i, face in enumerate(faces):
                # Obtenha as landmarks faciais
                landmarks = predictor(gray, face)

                # Calcule a bounding box alinhada para a face
                aligned_face_bbox = dlib.get_face_chip_details([dlib.full_object_detection(face, landmarks)], size=256)[0].rect

                # Recorte a face alinhada da imagem
                aligned_face = image[aligned_face_bbox.top():aligned_face_bbox.bottom(),
                                    aligned_face_bbox.left():aligned_face_bbox.right()]

                # Salve a face alinhada em um diretório temporário
                temp_face_path = os.path.join(output_dir, "temp", f"{filename}_face{i}.jpg")
                os.makedirs(os.path.dirname(temp_face_path), exist_ok=True)
                cv2.imwrite(temp_face_path, aligned_face)

                # Compare a face alinhada com as faces salvas nos subdiretórios do diretório de saída
                found_similar_face = False
                for subdir in os.listdir(output_dir):
                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.isdir(subdir_path) or subdir == "temp":
                        continue

                    # Compare a face alinhada com a primeira face em cada subdiretório
                    sample_face_path = os.path.join(subdir_path, os.listdir(subdir_path)[0])
                    sample_face = cv2.imread(sample_face_path)

                    if sample_face is not None:
                        if self.compare_faces(aligned_face, sample_face):
                            # Se uma face similar for encontrada, salve a face alinhada no subdiretório correspondente
                            final_face_path = os.path.join(subdir_path, f"{filename}_face{i}.jpg")
                            os.rename(temp_face_path, final_face_path)
                            found_similar_face = True
                            break

                if not found_similar_face:
                    # Se não encontrar nenhuma face similar, crie um novo subdiretório e salve a face alinhada nele
                    new_subdir_path = os.path.join(output_dir, f"face_group{len(os.listdir(output_dir))}")
                    os.makedirs(new_subdir_path, exist_ok=True)
                    final_face_path = os.path.join(new_subdir_path, f"{filename}_face{i}.jpg")
                    os.rename(temp_face_path, final_face_path)

        # Remova o diretório temporário
        if os.path.exists(os.path.join(output_dir, "temp")):
            shutil.rmtree(os.path.join(output_dir, "temp"))



    def compare_faces(self, face1, face2, threshold=0.6):
        """Compara duas faces usando o modelo de reconhecimento facial."""
        # Calcule as descrições das faces
        face1_descriptor = self.facerec.compute_face_descriptor(face1)
        face2_descriptor = self.facerec.compute_face_descriptor(face2)

        # Calcule a distância euclidiana entre as descrições das faces
        distance = np.linalg.norm(np.array(face1_descriptor) - np.array(face2_descriptor))

        # Retorne True se a distância for menor que o limite, caso contrário, retorne False
        return distance < threshold


    def create_widgets(self):
        """Cria os widgets para a janela principal."""
        self.input_dir_label = tk.Label(self.root, text='Diretório de entrada das fotos:')
        self.input_dir_entry = tk.Entry(self.root, textvariable=self.input_dir_var)
        self.input_dir_button = tk.Button(self.root, text='Selecionar', command=self.select_input_dir)

        self.output_dir_label = tk.Label(self.root, text='Diretório de saída das fotos:')
        self.output_dir_entry = tk.Entry(self.root, textvariable=self.output_dir_var)
        self.output_dir_button = tk.Button(self.root, text='Selecionar', command=self.select_output_dir)

        self.faces_dir_label = tk.Label(self.root, text='Diretório de treinamento das faces:')
        self.faces_dir_entry = tk.Entry(self.root, textvariable=self.faces_dir_var)
        self.faces_dir_button = tk.Button(self.root, text='Selecionar', command=self.select_faces_dir)

        self.run_button = tk.Button(self.root, text='Executar', command=self.run)


    def arrange_widgets(self):
        """Organiza os widgets na janela principal."""
        self.input_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.input_dir_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.input_dir_button.grid(row=0, column=2, padx=10, pady=10)

        self.output_dir_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.output_dir_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.output_dir_button.grid(row=1, column=2, padx=10, pady=10)

        self.faces_dir_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.faces_dir_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.faces_dir_button.grid(row=2, column=2, padx=10, pady=10)

        self.run_button.grid(row=3, column=1, padx=10, pady=10)


    def mainloop(self):
        """Inicia o loop principal da janela."""
        self.root.mainloop()


if __name__ == '__main__':
    app = PhotoSeparatorGUI()
    app.mainloop()