import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import dlib
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading


class PhotoSeparatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Separador de Fotos por Reconhecimento Facial')

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.faces_dir_var = tk.StringVar()

        self.lock = threading.Lock()

        self.create_widgets()
        self.arrange_widgets()

        self.facerec = dlib.face_recognition_model_v1(os.path.join('dlib_face_recognition_resnet_model_v1.dat'))

    def select_input_dir(self):
        input_dir = filedialog.askdirectory()
        if input_dir:
            self.input_dir_var.set(input_dir)
            print('Diretório de entrada selecionado:', input_dir)

    def select_output_dir(self):
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_var.set(output_dir)
            print('Diretório de saída selecionado:', output_dir)

    def run(self):
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        faces_dir = os.path.dirname(os.path.realpath(__file__))

        if not self.check_directories(input_dir, output_dir, faces_dir):
            return
        
        self.separate_photos(input_dir, output_dir, faces_dir)

    def print_error(self, title, message):
        messagebox.showerror(title, message)
        print(f'{title}: {message}')

    def check_directories(self, input_dir, output_dir, faces_dir):
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

    def process_image(self, input_file_path, file, face_dict, detector, sp, output_dir, folder_counter):
        img = dlib.load_rgb_image(input_file_path)
        dets = detector(img, 1)

        if len(dets) > 0:
            full_detections = dlib.full_object_detections()

            for k, d in enumerate(dets):
                shape = sp(img, d)
                full_detection = dlib.full_object_detection(d, shape.parts())
                full_detections.append(full_detection)

            for detection in full_detections:
                face_chip_150 = dlib.get_face_chip(img, detection, size=150, padding=0.25)
                found_similar_face = False
                for folder_name, reference_face in face_dict.items():
                    if self.compare_faces(reference_face, face_chip_150):
                        found_similar_face = True
                        break

                if not found_similar_face:
                    with self.lock:
                        folder_counter += 1
                        folder_name = f'Person_{folder_counter}'
                        face_dict[folder_name] = face_chip_150

                person_folder_path = os.path.join(output_dir, folder_name)
                if not os.path.exists(person_folder_path):
                    os.makedirs(person_folder_path)

                output_file_path = os.path.join(person_folder_path, f"{os.path.splitext(file)[0]}_face_{detection.rect.left()}_{detection.rect.top()}{os.path.splitext(file)[1]}")
                dlib.save_image(face_chip_150, output_file_path)
                print('\033[1;49;32m' + f'Face found in {file}!!' + '\033[m')
        else:
            print('\033[1;49;31m' + f"No face detected in {file}" + '\033[m')

        return folder_counter, {folder_name: face_chip_150}


    def separate_photos(self, input_dir, output_dir, faces_dir):
        """Função principal para gerenciar a separação de fotos"""

        # Carregue o detector de faces, o shape predictor e o modelo de reconhecimento facial
        detector = dlib.get_frontal_face_detector()
<<<<<<< HEAD
        sp = dlib.shape_predictor(os.path.join(faces_dir, 'shape_predictor_68_face_landmarks.dat'))

        face_dict = {}
        folder_counter = 0

        for root, dirs, files in os.walk(input_dir):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_image, os.path.join(root, file), file, face_dict, detector, sp, output_dir, folder_counter) for file in files if file.endswith((".jpg", ".jpeg", ".png"))]
                for future in concurrent.futures.as_completed(futures):
                    updated_folder_counter, updated_face_dict = future.result()
                    with self.lock:  # Adquire o Lock
                        folder_counter = max(folder_counter, updated_folder_counter)
                        face_dict.update(updated_face_dict)
=======
        predictor = dlib.shape_predictor(os.path.join(faces_dir, "shape_predictor_68_face_landmarks.dat"))
        self.facerec = dlib.face_recognition_model_v1(os.path.join(faces_dir, "dlib_face_recognition_resnet_model_v1.dat"))

        # Itere sobre todas as imagens no diretório de entrada
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if not os.path.isfile(file_path):
                continue

            # Carregue a imagem usando o OpenCV
            image = cv2.imread(file_path)
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
                aligned_face_bbox = dlib.get_face_chip_details([landmarks], size=256)[0].rect

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
        shutil.rmtree(os.path.join(output_dir, "temp"))
>>>>>>> parent of 020503d (Update photo_separator_gui.py)


    def compare_faces(self, face1, face2, threshold=0.6):
        face1_descriptor = self.facerec.compute_face_descriptor(face1)
        face2_descriptor = self.facerec.compute_face_descriptor(face2)

        distance = np.linalg.norm(np.array(face1_descriptor) - np.array(face2_descriptor))
        return distance < threshold

    def create_widgets(self):
        self.input_dir_label = tk.Label(self.root, text='Diretório de entrada das fotos:')
        self.input_dir_entry = tk.Entry(self.root, textvariable=self.input_dir_var)
        self.input_dir_button = tk.Button(self.root, text='Selecionar', command=self.select_input_dir)

        self.output_dir_label = tk.Label(self.root, text='Diretório de saída das fotos:')
        self.output_dir_entry = tk.Entry(self.root, textvariable=self.output_dir_var)
        self.output_dir_button = tk.Button(self.root, text='Selecionar', command=self.select_output_dir)

        self.run_button = tk.Button(self.root, text='Executar', command=self.run)

    def arrange_widgets(self):
        self.input_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.input_dir_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.input_dir_button.grid(row=0, column=2, padx=10, pady=10)

        self.output_dir_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.output_dir_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.output_dir_button.grid(row=1, column=2, padx=10, pady=10)

        self.run_button.grid(row=3, column=1, padx=10, pady=10)

    def mainloop(self):
        """Inicia o loop principal da janela."""
        self.root.mainloop()


if __name__ == '__main__':
    app = PhotoSeparatorGUI()
    app.mainloop()