from tensorflow import keras #importa a biblioteca keras do tensorflow
import pandas

#* ------------------------------------------------------------------------------------------------------------------------------------- MLP
#* Criação do modelo Perceptron
#*------------------------------------- Estrutura -------------------------------------- #
modelo = keras.Sequential([keras.layers.Dense(units=1,input_shape=[2],name='neuronio')])
    # "keras.Sequential" cria uma rede neural unindo camadas
    # "keras.layers.Dense" é uma camada densa (ou seja, uma camada de rede neural comum)
    # "units=1" define a quantidade de neuronios na camada
    # "input_shape=[2]" define a quantidade de entradas da camada
    # "name='neuronio'" define o nome da camada

modelo.summary()
    # "modelo.summary()" imprime as informações do modelo, como a quantidade de neuronios, entradas e saídas de cada camada

keras.utils.plot_model(modelo,show_shapes=True)
    # "keras.utils.plot_model(modelo,show_shapes=True)" imprime o modelo em uma imagem, mostrando as entradas e saídas de cada camada
    # O modelo é uma rede neural simples com uma camada de entrada de 2 neuronios




#* ---------------------------------- Pesos e Viéses ---------------------------------- #
modelo.layers
    # "modelo.layers" é uma lista das camadas do modelo

modelo.layers[0].get_weights()
    # "modelo.layers[0].get_weights()" retorna as pesos e bias da camada "
    # "modelo.layers[0]" é a primeira camada do modelo

pesos,bias = modelo.layers[0].get_weights()
print(pesos.shape)
pesos
    # "pesos,bias = modelo.layers[0].get_weights()" retorna as pesos
    # "pesos.shape" imprime a forma dos pesos
    # "pesos" imprime os pesos

print(bias.shape)
bias
    # "bias.shape" imprime a forma do bias
    # "bias" imprime o bias




#* ----------------------------------- Inicializando Pesos e Bias ----------------------------------- #
modelo = keras.Sequential([keras.layers.Dense(units=1,input_shape=[2],name='neuronio',                #inicializa os pesos e bias com valores aleatórios
                                              kernel_initializer = keras.initializers.RandomNormal(), #inicializa os pesos com valores aleatórios
                                              bias_initializer = keras.initializers.Ones())])         #inicializa o bias com 1

modelo.layers[0].get_weights()
    # "modelo.layers[0].get_weights()" retorna as pesos e bias da camada "
    # "modelo.layers[0]" é a primeira camada do modelo
    # "keras.initializers.RandomNormal()" inicializa os pesos com valores aleatórios
    # "keras.initializers.Ones()" inicializa os bias com 1







#* ---------------------------------------------------------------------------------------------------------------------------------------- importando dados
##### É possivel realizar a coleta desse dataset através do método `datasets` da biblioteca 'sklearn' #####
from sklearn import datasets #importa o conjunto de dados de iris

iris = datasets.load_iris(return_X_y = True) #carrega o conjunto de dados de iris, cria uma tupla

x = iris[0] #atribui os dados de entrada a variável x
y = iris[1] #atribui os dados de saída a variável y
print(x)    #imprime os dados de entrada

datasets.load_iris()['feature_names'] #retorna os nomes das variáveis de entrada

print(y)                              #imprime os dados de saída

datasets.load_iris()['target_names']  #retorna os nomes das variáveis de saída







#* ---------------------------------------------------------------------------------------------------------------------------------------- criando modelo
#* --------------------------- Gerando gráfico de distibuição de pétalas --------------------------- #
import matplotlib.pyplot as plt #importa a biblioteca de plotagem
import seaborn as sns           #importa a biblioteca de plotagem

sns.scatterplot(x = x[:,2], y = x[:,3], hue = y, palette = 'tab10') #gera um gráfico de dispersão com os dados de entrada
plt.xlabel('comprimento (m)',fontsize =16)                         #adiciona o título do eixo x
plt.ylabel('largura (m)', fontsize=16)                             #adiciona o título do eixo y
plt.title('Distribuição pétalas', fontsize = 18)                    #adiciona o título do gráfico
plt.show()                                                          #exibe o gráfico


#* --------------------------- Gerando modelo de distribuição de sépalas --------------------------- #
sns.scatterplot(x = x[:,0], y = x[:,1], hue = y, palette = "tab10") #gera um gráfico de dispersão com os dados de entrada
plt.xlabel('comprimeto (cm)', fontsize = 16)                        #adiciona o título do eixo x
plt.ylabel('largura (cm)', fontsize = 16)                           #adiciona o título do eixo y
plt.title('Distribuição sépalas', fontsize = 18)                    #adiciona o título do gráfico
plt.show()                                                          #exibe o gráfico






#* ------------------------------------------------------------------------------------------------------------------------------------------ tratamento
#* -------------------------------- Categorização -------------------------------- #
y.shape                           #retorna a dimensão dos dados de saída
y = keras.utils.to_categorical(y) #transforma os dados de saída em categorias
y.shape                           #retorna a dimensão dos dados de saída
print(y)                          #imprime os dados de saída


#* --------------------------------- Normalização -------------------------------- #
##### Os dados serão normalizados entre [0, 1], para isso utilizamos o método `MinMaxScaler` #####
from sklearn.preprocessing import MinMaxScaler #importa a biblioteca de normalização

scaler = MinMaxScaler()     #cria um objeto para normalização
x = scaler.fit_transform(x) #aplica a normalização nos dados de entrada


#* ----------------------------- Separação de conjunto --------------------------- #
##### A separação em conjuntos de treino e teste garantem um melhor processo de criação do modelo.          #####
##### Esses conjuntos são definidos a partir do conjunto total de dados, o qual separameos por proporções:  #####
##### 80% - Treino                                                                                          #####
##### 20% - Teste                                                                                           #####
from sklearn.model_selection import train_test_split #importa a biblioteca de separação de conjuntos

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 42) #divide os dados
x_treino.shape #retorna a dimensão do conjunto de treino
x_teste.shape  #retorna a dimensão do conjunto de teste







#* ------------------------------------------------------------------------------------------------------------------------------------------ Modelo MLP
#* ------------------------------- Criação do modelo ------------------------------- #
##### Fazemos um modelo MLP definido por 1 camada de entrada, 1 camada oculta e 1 camada de saída. #####
modelo = keras.Sequential([keras.layers.InputLayer(input_shape = [4,], name = 'entrada'),                        #define a camada de entrada
                           keras.layers.Dense(512, activation = 'relu', name = 'oculta',                         #define a camada oculta
                                              kernel_initializer = keras.initializers.RandomNormal(seed = 142)), #define o inicializador da camada oculta
                                              keras.layers.Dense(3, activation = 'softmax', name = 'saida')])    #define a camada de saída

modelo.summary() #retorna a estrutura do modelo







#* ------------------------------------------------------------------------------------------------------------------------------------------ Treinamento
#* ------------------------------ Compilando o modelo ------------------------------ #
##### Compilar o modelo se dá pelo método `compile`.                                                             #####
##### Nessa etapa devemos especificar a função de perda e o otimizador a ser usado.                              #####
##### Uma opção também é especificar as métricas a serem calculadas durante o treinamento e avaliação do modelo. #####

modelo.compile(loss = 'categorical_crossentropy',  #definie a função de perda |
               optimizer = 'rmsprop',              #define o otimizador       | #*compila o modelo
               metrics = ['categorical_accuracy']) #define a métrica          |

#Explicação:
#1. A perda escolhida foi a 'categorical_crossentropy' pois os resultados em y são valores categóricos.
#2. O uso do otimizador 'rmsprop' indica que iremos treinar o modelo baseado em gradiente descendente, que calcula a média móvel de gradientes quadrados para normalizar o gradiente (processo de retropropagação será aplicado ao realizar o treinamento).
#3. Por esse modelo ser um classificador simples, é interessante calcular o valor da acurácia durante o treino e validação.


#* ------------------------------------ Treino ------------------------------------- #
##### O treinamento é feito com o método `fit`, sendo especificado as entredas e saídas esperadas de treino, épocas e também os dados de validação. #####

#Explicação:
#1. Para treinamento é preciso passar os dados de entradas e saídas do conjunto de treinamento.
#2. Nesse caso o número de épocas também é essencial pois o padrão da biblioteca é 1 época, o que não é tão bom para o aprendizado.
#3. Passamos também a porcentagem do conjunto de validação para serem considerados do conjunto de treino - 30% do conjunto de treino

epocas = 4800                       #define o número de épocas
historico = modelo.fit(x_treino, y_treino,     #define as entradas e saídas de treino     |
                       epochs = epocas,        #define o número de épocas do modelo       | #*treina o modelo
                       validation_split = 0.3) #treina o modelo com as épocas e validação |

#*Interpretando a exibição:

#Exemplo de exibição: 

#Epoch 100/100
#3/3 [==============================] - 0s 47ms/step - loss: 0.1451 - categorical_accuracy: 0.9524 - val_loss: 0.1456 - val_categorical_accuracy: 0.9722

# - A época de treinamento: Epoch 200/200
# - Quantidade de instâncias processadas: 3/3
#   |Aqui, temos a quantidade de amostras divididas pela número de batch_size. Normalmente a batch é definida com 32 no keras, 
#   |temos 84 amostras (70% do conjunto de treino) no treinamento, resultando assim em aproximadamente 3 instâncias.
# - A barra de progresso: [==============================]
# - Tempo de treinamento de cada amostra: 0s 13ms/step
# - Perda e acurária no conjunto de treinamento: loss: 0.1506 - categorical_accuracy: 0.9524
# - Perda e acurária no conjunto de validação: val_loss: 0.1494 - val_categorical_accuracy: 0.9722






#* ------------------------------------------------------------------------------------------------------------------------------------------- Avaliação
#* ---------------------------------- Aprendizado ---------------------------------- #
##### Podemos avaliar o desempenho do nosso modelo durante o treinamento com os dados de 'historico' através do 'history' e plotar o processo de aprendizado #####

print(historico.history) #retorna o histórico de treinamento

#*Plotagem:
import pandas as pd                    #importa a biblioteca pandas
pd.DataFrame(historico.history).plot() #transforma o histórico em um DataFrame
plt.grid()                             #exibe a grade
plt.show()                             #exibe a plotagem do histórico de treinamento

#Para observar mais de perto o aprendizado do modelo podemos plotar curvas individuais e perceber como no aprendizado, ambos os conjuntos obtiveram resultados similares. Isso nos garante que não ocorreu sobreajuste no treinamento.
#A constancia das curvas é diferente pois os valores de erro da validação é calculada por época e para o treinamento é feito uma média dos valores de erro durante as iterações de cada época.

#*Cria duas subplots
fig, ax = plt.subplots(1, 2, figsize = (14, 5))

#*Plota a perda de treinamento e validação no primeiro subplot:
ax[0].plot(historico.history['loss'], color = '#111487', linewidth = 1.25, label = "Perda de treinamento")                       #perda de treinamento
ax[0].plot(historico.history['val_loss'], color = '#EFA316', linewidth = 1.25, label = "Perda da validação")                     #perda de validação
ax[0].set_xlabel('Épocas')                               #rótulo do eixo x
ax[0].set_ylabel('Perda')                                #rótulo do eixo y
ax[0].set_title('Perda durante o treinamento')           #título do subplot
ax[0].legend(loc = 'best', shadow = True)                #legenda

#*Plota a acurácia de treinamento e validação no segundo subplot:
ax[1].plot(historico.history['categorical_accuracy'], color = '#111487', linewidth = 1.25, label = "Acurácia de treinamento")    #acurácia de treinamento
ax[1].plot(historico.history['val_categorical_accuracy'], color = '#EFA316', linewidth = 1.25, label = "Acurácia de validação")  #acurácia de validação
ax[1].set_xlabel('Épocas')                               #rótulo do eixo x
ax[1].set_ylabel('Acurácia')                             #rótulo do eixo y
ax[1].set_title('Acurácia durante o treinamento')        #título do subplot
ax[1].legend(loc = 'best', shadow = True)                #legenda

plt.suptitle('Desempenho do treinamento', fontsize = 18) #título geral da figura
plt.show()                                               #exibe a plotagem


#* ------------------------------------------------------------------------------------------------------------------------------------------- Teste
##### Podemos testar o modelo e verificar seu resultado final através do método 'evaluate' que nos mostra a perda e acurácia obtida no conjunto de teste.
modelo.evaluate(x_teste, y_teste) #retorna a perda e acurácia no conjunto de teste


##### O método 'predict' gera a predição do modelo para as entradas enviadas.
import numpy as np #importa a biblioteca numpy

dados_entrada = np.array([0.61, 0.5, 0.69, 0.79]) #cria um array com as entradas
dados_entrada = dados_entrada.reshape(1, -1)      #transforma o array em uma matriz de 1 linha e n colunas
previsao = modelo.predict(dados_entrada)          #cria a previsão dos dados
print(previsao)                                   #retorna a previsão
print('')





#* -------------------------------------------------------------------------------------------------------------------------------------------- RESUMO DO CÓDIGO
# 1. Importação das bibliotecas necessárias
# 2. Carregamento dos dados
# 3. Preparação dos dados
# 4. Divisão dos dados em conjuntos de treinamento e teste
# 5. Criação do modelo
# 6. Compilação do modelo
# 7. Treinamento do modelo
# 8. Avaliação do modelo
# 9. Teste do modelo
# 10. Plotagem dos resultados
# 11. Previsão do modelo
# 12. Impressão da previsão
# 13. Impressão da perda e acurácia no conjunto de teste
# 14. Impressão da perda e acurácia no conjunto de validação
# 15. Impressão da perda e acurácia no conjunto de treinamento