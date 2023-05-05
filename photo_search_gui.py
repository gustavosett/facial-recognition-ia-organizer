import hashlib
from multiprocessing import Pool, cpu_count
import os
import shutil
import cv2
import dlib
from tqdm import tqdm
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pandas as pd

face_dict = {}
image_hashes = set()

class PhotoSearchGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Buscador de Fotos por Reconhecimento Facial')

        self.threshold_var = tk.DoubleVar(value=0.55)
        self.persons_dir_var = tk.StringVar()
        self.search_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        self.create_widgets()
        self.arrange_widgets()

        try:
            self.facerec = dlib.face_recognition_model_v1(
                os.path.join('dlib_face_recognition_resnet_model_v1.dat'))
            self.detector = dlib.get_frontal_face_detector()
            self.sp = dlib.shape_predictor(os.path.join(
                os.getcwd(), 'shape_predictor_68_face_landmarks.dat'))

        except Exception as e:
            self.print_error(
                'Erro ao carregar modelo de reconhecimento facial', str(e))

    def image_hash(self, image):
        return hashlib.md5(image).hexdigest()

    def check_reference_images(self, input_dir):
        """Verifica a qualidade das imagens de referência."""
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    input_file_path = os.path.join(root, file)
                    img = dlib.load_rgb_image(input_file_path)
                    dets = self.detector(img, 1)

                    if len(dets) == 0:
                        self.print_error(
                            'Imagem de referência ruim', f"A imagem de referência '{file}' não possui um rosto detectável. Por favor, use uma imagem de melhor qualidade.")
                        return False
        return True

    def compare_faces(self, face1, face2):
        """Compara duas faces usando o modelo de reconhecimento facial."""
        try:
            threshold = self.threshold_var.get()
            resized_face1 = cv2.resize(face1, (150, 150))
            resized_face2 = cv2.resize(face2, (150, 150))
            face1_descriptor = self.facerec.compute_face_descriptor(
                resized_face1)
            face2_descriptor = self.facerec.compute_face_descriptor(
                resized_face2)

            distance = np.linalg.norm(
                np.array(face1_descriptor) - np.array(face2_descriptor))
            return distance < threshold
        except Exception as e:
            self.print_error('Erro ao comparar faces', str(e))
            return False

    def check_reference_images(self, input_dir):
        """Verifica a qualidade das imagens de referência."""
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    input_file_path = os.path.join(root, file)
                    img = dlib.load_rgb_image(input_file_path)
                    dets = self.detector(img, 1)

                    if len(dets) == 0:
                        self.print_error(
                            'Imagem de referência ruim', f"A imagem de referência '{file}' não possui um rosto detectável. Por favor, use uma imagem de melhor qualidade.")
                        return False
        return True

    def select_persons_dir(self):
        try:
            persons_dir = filedialog.askdirectory()
            if persons_dir:
                self.persons_dir_var.set(persons_dir)
                print('Diretório de pessoas selecionado:', persons_dir)
        except Exception as e:
            self.print_error(
                'Erro ao selecionar diretório de pessoas', str(e))

    def select_search_dir(self):
        try:
            search_dir = filedialog.askdirectory()
            if search_dir:
                self.search_dir_var.set(search_dir)
                print('Diretório de busca selecionado:', search_dir)
        except Exception as e:
            self.print_error('Erro ao selecionar diretório de busca', str(e))

    def select_output_dir(self):
        try:
            output_dir = filedialog.askdirectory()
            if output_dir:
                self.output_dir_var.set(output_dir)
                print('Diretório de saída selecionado:', output_dir)
        except Exception as e:
            self.print_error('Erro ao selecionar diretório de saída', str(e))

    def run(self):
        persons_dir = self.persons_dir_var.get()
        search_dir = self.search_dir_var.get()
        output_dir = self.output_dir_var.get()

        if not self.check_directories(persons_dir, search_dir, output_dir):
            return

        try:
            threading.Thread(target=self.search_photos, args=(
                persons_dir, search_dir, output_dir)).start()
            print('Buscando...')
        except Exception as e:
            self.print_error('Erro ao executar busca de fotos', str(e))

    def process_image(self, args):
        file_path, persons_dir, output_dir = args
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            if self.image_hash(data) in image_hashes:
                return
            image_hashes.add(self.image_hash(data))

            img = dlib.load_rgb_image(file_path)

        except RuntimeError:
            print(f"Skipping {file} due to invalid image file.")
            return

        dets = self.detector(img, 1)
        if len(dets) == 0:
            return

        for det in dets:
            shape = self.sp(img, det)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)

            for person_name, reference_descriptor in face_dict.items():
                distance = np.linalg.norm(
                    np.array(face_descriptor) - np.array(reference_descriptor))

                if distance < self.threshold_var.get():
                    person_folder = os.path.join(output_dir, person_name)
                    if not os.path.exists(person_folder):
                        os.makedirs(person_folder)

                    output_file_path = os.path.join(person_folder, file)
                    # verifica se a foto já foi salva para evitar duplicatas
                    if not os.path.isfile(output_file_path):
                        cv2.imwrite(output_file_path, cv2.cvtColor(
                            np.array(img), cv2.COLOR_RGB2BGR))


    def print_error(self, title, message):
        messagebox.showerror(title, message)
        print(f'{title}: {message}')

    def check_directories(self, persons_dir, search_dir, output_dir):
        try:
            error_messages = [
                ('Por favor, selecione um diretório de pessoas.', persons_dir),
                ('Por favor, selecione um diretório de busca.', search_dir),
                ('Por favor, selecione um diretório de saída.', output_dir),
            ]

            for error_message, directory in error_messages:
                if not directory:
                    self.print_error('Erro', error_message)
                    return False

            if persons_dir == search_dir or persons_dir == output_dir or search_dir == output_dir:
                self.print_error(
                    'Erro', 'Os diretórios selecionados não podem ser iguais. Por favor, escolha diretórios diferentes.')
                return False

            for directory in [persons_dir, search_dir, output_dir]:
                if not os.path.isdir(directory):
                    self.print_error(
                        'Erro', f'O diretório "{directory}" não é válido. Por favor, selecione um diretório válido.')
                    return False
            return True
        except Exception as e:
            self.print_error('Erro ao verificar os diretórios', str(e))
            return False

    def search_photos(self, persons_dir, search_dir, output_dir):
        # verifica se a pasta de referência de pessoas existe
        if not os.path.exists(persons_dir):
            tk.messagebox.showerror("Directory not found", "The persons directory was not found.")
            return

        # verifica se a pasta de busca existe
        if not os.path.exists(search_dir):
            tk.messagebox.showerror("Directory not found", "The search directory was not found.")
            return

        # verifica se a pasta de saída existe, senão, a cria
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # lê todas as imagens de referência e cria os descritores faciais
        for root, dirs, files in os.walk(persons_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    img = dlib.load_rgb_image(file_path)
                    dets = self.detector(img, 1)

                    for det in dets:
                        shape = self.sp(img, det)
                        face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                        person_name = os.path.basename(root)
                        face_dict[person_name] = face_descriptor

        # Cria uma lista de todos os arquivos para processamento
        all_files = []
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    all_files.append((file_path, persons_dir, output_dir, self.detector, self.sp))

        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(process_image_function, all_files), total=len(all_files), ncols=70, desc="Processing Images"))


    def create_widgets(self):
        self.persons_dir_label = tk.Label(
            self.root, text='Diretório com fotos das pessoas:')
        self.persons_dir_entry = tk.Entry(
            self.root, textvariable=self.persons_dir_var)
        self.persons_dir_button = tk.Button(
            self.root, text='Selecionar', command=self.select_persons_dir)

        self.threshold_label = tk.Label(
            self.root, text='Sensibilidade da correspondência de rostos:')
        self.threshold_scale = tk.Scale(
            self.root, from_=0.1, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.threshold_var)

        self.search_dir_label = tk.Label(
            self.root, text='Diretório de busca:')
        self.search_dir_entry = tk.Entry(
            self.root, textvariable=self.search_dir_var)
        self.search_dir_button = tk.Button(
            self.root, text='Selecionar', command=self.select_search_dir)

        self.output_dir_label = tk.Label(
            self.root, text='Diretório de saída das fotos encontradas:')
        self.output_dir_entry = tk.Entry(
            self.root, textvariable=self.output_dir_var)
        self.output_dir_button = tk.Button(
            self.root, text='Selecionar', command=self.select_output_dir)
        self.run_button = tk.Button(
            self.root, text='Iniciar busca', command=self.run)

    def arrange_widgets(self):
        self.persons_dir_label.grid(
            row=0, column=0, sticky='w', padx=5, pady=5)
        self.persons_dir_entry.grid(
            row=0, column=1, sticky='we', padx=5, pady=5)
        self.persons_dir_button.grid(
            row=0, column=2, sticky='e', padx=5, pady=5)

        self.threshold_label.grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.threshold_scale.grid(row=4, column=1, sticky='we', padx=5, pady=5)

        self.search_dir_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.search_dir_entry.grid(
            row=1, column=1, sticky='we', padx=5, pady=5)
        self.search_dir_button.grid(
            row=1, column=2, sticky='e', padx=5, pady=5)

        self.output_dir_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.output_dir_entry.grid(
            row=2, column=1, sticky='we', padx=5, pady=5)
        self.output_dir_button.grid(
            row=2, column=2, sticky='e', padx=5, pady=5)

        self.run_button.grid(row=3, column=0, columnspan=3, pady=5)

    def mainloop(self):
        """Inicia o loop principal da janela."""
        self.root.mainloop()

def process_image_function(args):
    file_path, persons_dir, output_dir, detector, sp = args

    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    img = dlib.load_rgb_image(file_path)
    dets = detector(img, 1)
    if len(dets) == 0:
        return

    for det in dets:
        shape = sp(img, det)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        for person, reference_descriptor in face_dict.items():
            print(person)
            dist = np.linalg.norm(np.array(face_descriptor) - np.array(reference_descriptor))
            if dist < 0.6:
                print(f'confirmado é a pessoa: {person}')
                save_folder = os.path.join(output_dir, person)
                if not os.path.exists(save_folder):
                    print(f'criando diretório para: {person}')
                    os.makedirs(save_folder)

                file_name = os.path.basename(file_path)
                save_path = os.path.join(save_folder, file_name)
                shutil.copyfile(file_path, save_path)
                print('arquivo copiado')
                break


if __name__ == '__main__':
    app = PhotoSearchGUI()
    app.mainloop()
