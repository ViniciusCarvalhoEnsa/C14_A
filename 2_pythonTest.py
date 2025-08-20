import PIL.Image
import PIL                   #importa a biblioteca PIL
from tensorflow import keras #importa a biblioteca keras do tensorflow
import numpy as np           #importa a biblioteca numpy
import pathlib               #importa a biblioteca pathlib

url = 'C:/Users/vinic/Documents/VS_Code/Estudo_tarefas/IC/Keras/uvas'

data_dir = pathlib.Path(url) #define o diretório da pasta com as imagens

#* ---------------------------------------------------------------------------------- Usando o diretório
print('')
print(len(list(data_dir.glob('*/*.JPG')))) #mostra o número de imagens

subfolders = [f.name for f in data_dir.iterdir() if f.is_dir] #repetição que vai varrer a pasta e se o 'f' for um diretório ele retorna a informação
print(subfolders)                                             #mostra as subpastas
print('')

leafblight = list(data_dir.glob('LeafBLight/*')) #pega todos os itens da pasta 'LeafBLight' e os coloca na lista leafblight
img = PIL.Image.open(str(leafblight[0]))         #abre a primeira imagem da lista leafblight
img.show()                                       #mostra a imagem

for subfolder in subfolders:                     #repetição que vai varrer a lista de subpastas
    path = data_dir / subfolder                  #define o caminho da pasta
    images = list(path.glob('*.JPG'))            #pega todos os itens da pasta e os coloca na lista images
    print(f'{subfolder}: {len(images)} imagens') #mostra o número de imagens em cada subpasta

    if images:
        img = PIL.Image.open(str(images[0]))     #abre a primeira imagem da lista
        img_array = np.array(img)                #converte a imagem em um array numpy
        print(f'Dimensões da primeira imagem em {subfolder}: {img_array.shape}') #mostra as dimensões da imagem














print('')