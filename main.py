import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
import time

def contar_moedas(imagem):
    start_time = time.time()
    
    # Verificar se a biblioteca necessária está instalada
    try:
        imagem = Image.open(imagem)
        imagem = np.array(imagem)
    except ImportError:
        print("Erro: a biblioteca PIL (Pillow) não está instalada.")
        print("Tente instalar a biblioteca com o comando: pip install pillow")
        return

    # Carregar a imagem, converter para escala de cinza, desfocar e detectar bordas
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    desfocado = cv2.GaussianBlur(cinza, (15, 15), 0)
    bordas = cv2.Canny(desfocado, 30, 150)

    # Realizar uma dilatação e erosão para fechar as lacunas entre as bordas das moedas
    dilatado = cv2.dilate(bordas.copy(), None, iterations=2)
    erosao = cv2.erode(dilatado.copy(), None, iterations=1)

    # Encontrar contornos na imagem da erosão
    contornos = cv2.findContours(erosao.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]

    # Loop sobre os contornos e desenhar cada um deles
    for contorno in contornos:
        cv2.drawContours(imagem, [contorno], -1, (0, 255, 0), 2)

    # Salvar a imagem de saída
    cv2.imwrite("moedas4.10.tiff", imagem)

    # Retornar o número de moedas
    execution_time = (time.time() - start_time)/60
    print(f"Tempo de execução: {execution_time} minutos")
    return len(contornos)

# Testar a função
if os.path.exists('moedas4321.tiff'):
    print(contar_moedas('moedas4321.tiff'))
else:
    print("Erro: a imagem 'moedas1234.tiff' não foi encontrada.")
