import math
import numpy as np

np.random.seed()

class Pergunta:

    def __init__(self, coluna, valor):
        self.coluna = coluna
        self.valor = valor

    def ehigual(self, exemplo):
        # Compare o atributo da pergunta com um exemplo dado
        val = exemplo[self.coluna]
        return val == self.valor

class Folha:
    "Esse nó contem apenas um valor, indicando a predição da classe quando esse nó"

    def __init__(self, rows):
        self.predicoes = class_counts(rows)
        
class Decision_Tree:
    #Utilizado para gerar uma árvore binária. Atributos: Arvore_True, Árvore_False e pergunta

    def __init__(self, pergunta, arvore_true, arvore_false):
        self.pergunta = pergunta
        self.arvore_true = arvore_true
        self.arvore_false = arvore_false
        
def unique_vals(rows, col):
    return set((row[col] for row in rows))

def class_counts(dataset):
    counts = {}
    for data in dataset:
        
        label = data[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#calcula de entropia
def entropia(rows):
    
    entropia = 0
    classes = class_counts(rows)
    for label in classes:
        prob = classes[label]/float(len(rows))
        entropia -= prob*(math.log2(prob))
    return entropia

def ganho_informacao(esq, direita, entropiaTotal):
    
    proporcao_esq = float(len(esq))/ (len(esq)+len(direita))
    return entropiaTotal - proporcao_esq * entropia(esq) - (1 - proporcao_esq) * entropia(direita)

def divide_dataset(dataset, pergunta):

    verdadeiras, falsas = [], []
    for inst in dataset:
        if pergunta.ehigual(inst):
            verdadeiras.append(inst)
        else:
            falsas.append(inst)
    return verdadeiras, falsas

def encontra_melhor_atributo(dataset):
    #Encontra melhor pergunta com base no ganho de informação
    melhor_ganho = 0  
    melhor_pergunta = None
    entropia_atual = entropia(dataset)
    numColunas = len(dataset[0]) - 1  # numero de colunas

    for coluna in range(0, numColunas):

        valores = unique_vals(dataset, coluna)

        for val in valores:

            pergunta = Pergunta(coluna, val)

            # 2º Realização da divisão do dataset
            positivas, falsas = divide_dataset(dataset, pergunta)

            # Se positivas == 0 ou falsas == 0, então a divisão é desconsiderada
            if len(positivas) == 0 or len(falsas) == 0:
                continue

            # Calcula o ganho de informação desse atributo
            ganho = ganho_informacao(positivas, falsas, entropia_atual)

            if ganho > melhor_ganho:
                melhor_ganho, melhor_pergunta = ganho, pergunta

    return melhor_ganho, melhor_pergunta

def constroi_arvore(dataset):

    ganho, pergunta = encontra_melhor_atributo(dataset)

    # Caso de parada: quando o ganho de informação é 0, ou seja, não há impurezza
    if ganho == 0:
        return Folha(dataset)

    # Divisão das instâncias
    instancias_verdadeiras, instancias_falsas = divide_dataset(dataset, pergunta)

    # Recursão para gerar a árvore da direita(Sempre TRUE)
    arvore_dir = constroi_arvore(instancias_verdadeiras)

    # Recursão para gerar a árvore da esquerda(sempre FALSE)
    arvore_esq = constroi_arvore(instancias_falsas)

    # Returna a árvore de decisão
    return Decision_Tree(pergunta, arvore_dir, arvore_esq)


def classificacao(instancia, decision_tree):
    
    # Caso base: ao alcançar um nó folha
    if isinstance(decision_tree, Folha):
        return decision_tree.predicoes
    
    if decision_tree.pergunta.ehigual(instancia):
        return classificacao(instancia, decision_tree.arvore_true)
    else:
        return classificacao(instancia, decision_tree.arvore_false)
    
#Recebimento de entradas

entrada = input().rstrip()
n, m = entrada.split(" ") #n - numero de registros m - número de atributos
n = int(n)
m = int(m)

training_data = []
inst_teste = []

for i in range(n-1):
    entrada = input().rstrip()
    registro = entrada.split(" ")
    training_data.append(registro)
    
for i in range(m):
    entrada = input().rstrip()
    inst = entrada.split(" ")
    inst_teste.append(inst)


########################################
######## INICIO DA RANDOM FOREST ###########
#########################################

#Técnica de Pasting sem reposição - Criar 3 datasets para treinar 3 árvores diferentes, usando 80% da base de treinamento - 11 amostras
pasted_data = []
for i in range(0,3):
    aux = training_data[:]
    num_amostras = int(len(training_data)*0.8)
    data = []
    for j in range(num_amostras):
        num_linha = np.random.randint(low = 0, high = len(aux))
        data.append(aux[num_linha])
        aux.pop(num_linha)
    pasted_data.append(data)
    
#Criação da Random Forest
floresta = list()

for dataset in pasted_data:
    arvore = constroi_arvore(dataset)
    floresta.append(arvore)

# Votação

for inst in inst_teste:
    
    numeros_sim = 0
    numeros_nao = 0
    
    for arvore in floresta:
        valor = list(classificacao(inst, arvore))[0]
        if valor == 'sim':
            numeros_sim += 1
        else:
            numeros_nao += 1
    
    if numeros_sim > numeros_nao:
        print("sim")
    else:
        print("nao")