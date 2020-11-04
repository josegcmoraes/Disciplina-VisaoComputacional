# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets, metrics, preprocessing, svm, naive_bayes, neighbors, model_selection
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from skimage import data,img_as_float,measure
import matplotlib.pylab as plt
import imageio
import os

diretorio=os.listdir('data/mpeg7/')
print(diretorio)
print("________")
im=[]
matriz_caracteristica=[]
vetor_rotulos=[]
classe=0
#i=0

for pasta in diretorio:
    imagens_pasta=os.listdir('data/mpeg7/'+pasta)
    print(imagens_pasta)
    print("________")
    
    for imagem in imagens_pasta:
        print(imagem)
        img=imageio.imread('data/mpeg7/'+pasta+'/'+imagem)
        plt.figure()
        plt.imshow(img)
        plt.show()
        
        #img=img_as_float(img)
        im.append(img)
        props=measure.regionprops(img,img)
        matriz_caracteristica.append([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity])
        #matrizcaracteristica = np.array([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity])
        
        #i=i+1
        vetor_rotulos.append(classe)
    classe=classe+1

print('________________________________________________')    
print("matriz de característica")
print(matriz_caracteristica)
print("target")
print(vetor_rotulos)

x=matriz_caracteristica
y=vetor_rotulos


x_treino, x_teste, y_treino, y_teste = model_selection.train_test_split(matriz_caracteristica, vetor_rotulos, test_size=0.25, random_state =42)


# Normalização das características
#x_treino = preprocessing.scale(x_treino)
#x_teste = preprocessing.scale(x_teste)
scaler = StandardScaler()
scaler.fit(x_treino)
x_treino=scaler.transform(x_treino)
scaler.fit(x_teste)
x_teste=scaler.transform(x_teste)
print('________________________________________________')
print('Valores normalizados x_treino')
print(x_treino)
print('Valores normalizados x_teste')
print(x_teste)



#__________________________________________
#clf = Perceptron(n_iter=20, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
clf = neural_network.MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42,verbose=10)
clf.fit(x_treino,y_treino)
pred = clf.predict(x_teste)

print('\n\n________________________________________________')
print('Classificador Perceptron...........')
print('Predição:') 
print(pred)
print('\nReal:') 
print(y_teste)
print('\nMatriz de confusão:') 
print(metrics.confusion_matrix(y_teste, pred))
print('\nRelatório de classificação:') 
print(metrics.classification_report(y_teste, pred))
print('acurácia')
print(metrics.accuracy_score(y_teste, pred))
