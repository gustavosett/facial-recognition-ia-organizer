
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
import cv2
import os
import shutil


def select_input_dir():
    input_dir = filedialog.askdirectory()
    if input_dir:
        input_dir_var.set(input_dir)
        print('Diretório de entrada selecionado:', input_dir)

def main():
    root = tk.Tk()
    root.title('Separador de Fotos por Reconhecimento Facial')

    # Define as variáveis input_dir_var, output_dir_var e faces_dir_var
    input_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    faces_dir_var = tk.StringVar()

    # Cria os widgets Entry e Label para as variáveis
    input_dir_label = tk.Label(root, text='Diretório de entrada das fotos:')
    input_dir_entry = tk.Entry(root, textvariable=input_dir_var)
    output_dir_label = tk.Label(root, text='Diretório de saída das fotos:')
    output_dir_entry = tk.Entry(root, textvariable=output_dir_var)
    faces_dir_label = tk.Label(root, text='Diretório de treinamento das faces:')
    faces_dir_entry = tk.Entry(root, textvariable=faces_dir_var)

    # Cria os botões para selecionar os diretórios
    input_dir_button = tk.Button(root, text='Selecionar', command=select_input_dir)
    output_dir_button = tk.Button(root, text='Selecionar', command=select_output_dir)
    faces_dir_button = tk.Button(root, text='Selecionar', command=select_faces_dir)

    # Cria o botão para executar a separação de fotos
    run_button = tk.Button(root, text='Executar', command=run)

    # Define a geometria dos widgets na janela
    input_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    input_dir_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W+tk.E)
    input_dir_button.grid(row=0, column=2, padx=10, pady=10)

    output_dir_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    output_dir_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W+tk.E)
    output_dir_button.grid(row=1, column=2, padx=10, pady=10)

    faces_dir_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    faces_dir_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W+tk.E)
    faces_dir_button.grid(row=2, column=2, padx=10, pady=10)

    run_button.grid(row=3, column=1, padx=10, pady=10)

    root.mainloop()
main()


