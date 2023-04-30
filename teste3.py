import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
import cv2
import os
import shutil

# Função para selecionar o diretório de entrada das fotos
def select_input_dir():
    input_dir = filedialog.askdirectory()
    if input_dir:
        input_dir_var.set(input_dir)

# Função para selecionar o diretório de saída das fotos
def select_output_dir():
    output_dir = filedialog.askdirectory()
    if output_dir:
        output_dir_var.set(output_dir)

# Função para selecionar o diretório com as imagens de treinamento
def select_faces_dir():
    faces_dir = filedialog.askdirectory()
    if faces_dir:
        faces_dir_var.set(faces_dir)

# Função para executar a separação de fotos por reconhecimento facial
def run():
    input_dir = input_dir_var.get()
    output_dir = output_dir_var.get()
    faces_dir = faces_dir_var.get()

    # Verifica se os diretórios foram selecionados
    if not input_dir:
        messagebox.showerror('Erro', 'B:\Separação\Entrada das fotos')
        return
    if not output_dir:
        messagebox.showerror('Erro', 'B:\Separação\Saida das Fotos')
        return
    if not faces_dir:
        messagebox.showerror('Erro', 'B:\Separação\Treinamento')
        return

    # Cria o detector de faces e o reconhecedor facial
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    # Carrega as imagens de treinamento e cria um modelo de reconhecimento facial
    known_faces = []
    known_names = []
    for file_name in os.listdir(faces_dir):
        image = cv2.imread(os.path.join(faces_dir, file_name))
        face_rects = detector(image, 1)
        if len(face_rects) != 1:
            print('A imagem de treinamento {} não contém uma única face'.format(file_name))
        else:
            shape = predictor(image, face_rects[0])
            face_descriptor = facerec.compute_face_descriptor(image, shape)
            known_faces.append(face_descriptor)
            known_names.append(os.path.splitext(file_name)[0])

    # Percorre todas as imagens no diretório de entrada
    for file_name in os.listdir(input_dir):
        image = cv2.imread(os.path.join(input_dir, file_name))
        face_rects = detector(image, 1)
        if len(face_rects) == 0:
            print('Não foi possível detectar nenhuma face na imagem {}'.format(file_name))
        else:
            for face_rect in face_rects:
                shape = predictor(image, face_rect)
                face_descriptor = facerec.compute_face_descriptor(image, shape)

                # Compara a face com as imagens de treinamento para identificar a pessoa
                #min_distance = float('inf')
                #in_index =
                                # Compara a face com as imagens de treinamento para identificar a pessoa
                min_distance = float('inf')
                min_index = -1
                for i, known_face in enumerate(known_faces):
                    distance = 0
                    for j in range(len(face_descriptor)):
                        distance += (face_descriptor[j] - known_face[j]) ** 2
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i

                # Salva a imagem na pasta correspondente à pessoa identificada
                name = 'Desconhecido'
                if min_index != -1 and min_distance < 0.6:
                    name = known_names[min_index]
                output_path = os.path.join(output_dir, name)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                shutil.copy(os.path.join(input_dir, file_name), output_path)

                # Mostra a imagem com a identificação da pessoa
                cv2.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
                cv2.putText(image, name, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Resultado', image)
                cv2.waitKey(0)

    # Fecha a janela da aplicação
    root.destroy()

    # Cria a janela da aplicação
    root = tk.Tk()
    root.title('Separador de Fotos por  Reconhecimento Facial')

    # Cria os widgets da janela
    input_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    faces_dir_var = tk.StringVar()

    input_dir_label = tk.Label(root, text='Diretório de entrada das fotos:')
    input_dir_entry = tk.Entry(root, textvariable=input_dir_var)
    input_dir_button = tk.Button(root, text='Selecionar', command=select_input_dir)

    output_dir_label = tk.Label(root, text='Diretório de saída das fotos:')
    output_dir_entry = tk.Entry(root, textvariable=output_dir_var)
    output_dir_button = tk.Button(root, text='Selecionar', command=select_output_dir)

    faces_dir_label = tk.Label(root, text='Diretório com as imagens de treinamento:')
    faces_dir_entry = tk.Entry(root, textvariable=faces_dir_var)
    faces_dir_button = tk.Button(root, text='Selecionar', command=select_faces_dir)

    run_button = tk.Button(root, text='Executar', command=run)

    # Posiciona os widgets na janela
    input_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    input_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky='we')
    input_dir_button.grid(row=0, column=2, padx=5, pady=5)

    output_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    output_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky='we')
    output_dir_button.grid(row=1, column=2, padx=5, pady=5)

    faces_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    faces_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky='we')
    #aces_dir_button.grid(row=

    input_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    faces_dir_var = tk.StringVar()

    input_dir_label = tk.Label(root, text='Diretório de entrada das fotos:')
    input_dir_entry = tk.Entry(root, textvariable=input_dir_var)
    input_dir_button = tk.Button(root, text='Selecionar', command=select_input_dir)

    output_dir_label = tk.Label(root, text='Diretório de saída das fotos:')
    output_dir_entry = tk.Entry(root, textvariable=output_dir_var)
    output_dir_button = tk.Button(root, text='Selecionar', command=select_output_dir)

    faces_dir_label = tk.Label(root, text='Diretório com as imagens de treinamento:')
    faces_dir_button = tk.Button(root, text='Selecionar', command=select_faces_dir)

    run_button = tk.Button(root, text='Executar', command=run)
    input_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    input_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    input_dir_button.grid(row=0, column=2, padx=5, pady=5)

    output_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    output_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    output_dir_button.grid(row=1, column=2, padx=5, pady=5)

    faces_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    faces_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
    faces_dir_button.grid(row=2, column=2, padx=5, pady=5)

    run_button.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

    run_button = tk.Button(root, text='Executar', command=run)
    input_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    input_dir_entry.grid(row=0, column=1, padx=5, pady=5)
    input_dir_button.grid(row=0, column=2, padx=5, pady=5)

    output_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    output_dir_entry.grid(row=1, column=1, padx=5, pady=5)
    output_dir_button.grid(row=1, column=2, padx=5, pady=5)

    faces_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    faces_dir_entry.grid(row=2, column=1, padx=5, pady=5)
    faces_dir_button.grid(row=2, column=2, padx=5, pady=5)

    run_button.grid(row=3, column=1, padx=5, pady=5)

    root.mainloop()