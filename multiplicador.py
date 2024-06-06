from PIL import Image
import numpy as np

# Abra a imagem
img = Image.open('C:\Users\aluno\Documents\Teste_Bilhao\moedas.png')

# Redimensione a imagem para 100x100
img = img.resize((100, 100))

# Converta a imagem para uma matriz numpy
img_array = np.array(img)

# Multiplique a imagem por ela mesma
img_multiplied = img_array * img_array

# Converta a matriz numpy de volta para uma imagem
img_multiplied = Image.fromarray(img_multiplied)

# Salve a imagem multiplicada
img_multiplied.save('C:\Users\aluno\Documents\Teste_Bilhao')
