# Separador de Fotos por Reconhecimento Facial

Este software utiliza técnicas de reconhecimento facial para separar fotos de diferentes indivíduos em pastas distintas, facilitando a organização de grandes volumes de imagens.

## Funcionalidades

- Seleção de diretório de entrada com as fotos a serem organizadas.
- Seleção de diretório de saída onde as fotos organizadas serão salvas em subpastas.
- Utiliza IA para reconhecimento facial e agrupamento das imagens.
- Interface gráfica amigável para fácil utilização.

## Pré-requisitos

- Python 3.x
- Bibliotecas: Tkinter, dlib, cv2 (OpenCV), numpy

## Como utilizar

1. Execute o software.
2. Selecione o diretório de entrada com as fotos a serem organizadas.
3. Selecione o diretório de saída onde as fotos organizadas serão salvas.
4. Clique em "Executar" e aguarde enquanto o software organiza as fotos.

## Instalação das dependências

Instale as dependências utilizando pip:

```
pip install dlib opencv-python numpy tk
```

## Estrutura do Projeto

```
/photo_separator_gui.py    - Script principal contendo a lógica e a interface gráfica.
/test_photo_separator.py   - Testes unitários para o software.
/face_samples/             - (Opcional) Amostras de faces para testes.
```

## Como contribuir

1. Faça um fork do projeto.
2. Crie uma nova branch para suas modificações.
3. Envie um pull request.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Autores

- @gustavosett

---