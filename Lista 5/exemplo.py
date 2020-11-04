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
dados=[]
rotulos=[]
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
        
        im.append(img)
        props=measure.regionprops(img,img)
        dados.append([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity])
        #matrizcaracteristica = np.array([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity])
        
        #i=i+1
        rotulos.append(classe)
    classe=classe+1
print("dados")
print(dados)
print("rotulos")
print(rotulos)