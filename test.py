import tkinter as tk
from tkinter import filedialog


class PhotoSeparatorGUI:
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
        # Implemente o código de sep. de fotos aqui.
        pass

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
        self.faces_dir_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.faces_dir_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W + tk.E)
        self.faces_dir_button.grid(row=2, column=2, padx=10, pady=10)

        self.run_button.grid(row=3, column=1, padx=10, pady=10)

    def mainloop(self):
        """Inicia o loop principal da janela."""
        self.root.mainloop()


if __name__ == 'main':
    app = PhotoSeparatorGUI()
    app.mainloop()