from PIL import Image

def replicate_image(image_path, n, output_path):
    img = Image.open(image_path)
    width, height = img.size
    
    # Aumenta as dimensões da imagem
    new_width = width * n
    new_height = height * n
    
    # Cria uma nova imagem vazia
    new_img = Image.new("RGB", (new_width, new_height))
    
    # Preenche a nova imagem com cópias da imagem original
    for i in range(n):
        for j in range(n):
            new_img.paste(img, (i * width, j * height))
    
    # Salva a imagem após todos os blocos serem adicionados
    new_img.save(output_path, format="TIFF", compression="tiff_lzw")

# Solicita ao usuário o valor de n
while True:
    try:
        n = int(input("Digite o valor de n para replicação da imagem: "))
        break
    except ValueError:
        print("Por favor, insira um número inteiro válido.")

# Caminho da imagem
image_path = input("Digite o caminho da imagem JPG: ")

# Caminho de saída para a nova imagem
output_path = input("Digite o caminho de saída para a nova imagem TIFF: ")

# Chama a função para replicar a imagem
replicate_image(image_path, n, output_path)


