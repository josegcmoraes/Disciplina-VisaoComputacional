# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
import matplotlib.pylab as plt
from sklearn import datasets, metrics
from sklearn import svm, naive_bayes, neighbors, model_selection
from matplotlib.colors import ListedColormap


iris=datasets.load_iris()
x=iris.data
y=iris.target
#x=x[:,[1,3]]
#y[y==1]=0
#y[y==3]=1

x_treino,x_teste,y_treino,y_teste=model_selection.train_test_split(x,y,test_size=0.20)

print(x_treino)
skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
i = 0
for train_index, test_index in skf.split(x, y):
    print('Fold ' + str(i))
    x_treino, x_teste = x[train_index], x[test_index]
    y_treino, y_teste = y[train_index], y[test_index]
    
    print('Índices:')
    print('Treino: ' + str(train_index))
    print('Teste: ' + str(test_index))
    print('\nRótulos:')
    print('Treino: ' + str(y_treino))
    print('Teste: ' + str(y_teste))
    
    #print(x_treino)
    #print(x_teste)
    i=i+1
    print()
    
x_treino, x_teste, y_treino, y_teste = model_selection.train_test_split(iris.data, iris.target, test_size=0.33, random_state =42)
x_treino = x_treino[:,:2] 
x_teste = x_teste[:,:2] 

clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(x_treino, y_treino)
        
#________________________________________________________________
map_a = ListedColormap(['r', 'g'])
map_b = ListedColormap(['y', 'b']) 
plt.figure(figsize=(7,7))
# Plot also the training points 
plt.scatter(x_treino[:, 0], x_treino[:, 1], c=y_treino, cmap=map_a, edgecolor='k', s =80) 
plt.title("Classificador K-NN")
# Plot also the training points 
plt.scatter(x_teste[:, 0], x_teste[:, 1], c=y_teste, cmap=map_a, edgecolor='k', marker='^',  s=80) 
plt.title("Classificador K-NN")


clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(x_treino, y_treino) 
pred = clf.predict(x_teste)
print('\nClassificador K-NN..................................')
print('\n Predição:') 
print(pred) 
print('\nReal:') 
print(y_teste)
print('\nMatriz de confusão:') 
print(metrics.confusion_matrix(y_teste, pred))
print('\nRelatório de classificação:') 
print(metrics.classification_report(y_teste, pred))
print('acurácia')
print(metrics.accuracy_score(y_teste, pred))

#________________________________________________________________
clf = naive_bayes.GaussianNB()
clf.fit(x_treino, y_treino) 
pred = clf.predict(x_teste)
print('\nClassificador Bayes.................................')
print('\n Predição:') 
print(pred) 
print('\nReal:') 
print(y_teste)
print('\nMatriz de confusão:') 
print(metrics.confusion_matrix(y_teste, pred))
print('\nRelatório de classificação:') 
print(metrics.classification_report(y_teste, pred))
print('acurácia')
print(metrics.accuracy_score(y_teste, pred))







#________________________________________________________________
clf = svm.SVC()
clf.fit(x_treino, y_treino) 
pred = clf.predict(x_teste)
print('\nClassificador SVM.................................')
print('\n Predição:') 
print(pred) 
print('\nReal:') 
print(y_teste)
print('\nMatriz de confusão:') 
print(metrics.confusion_matrix(y_teste, pred))
print('\nRelatório de classificação:') 
print(metrics.classification_report(y_teste, pred))
print('acurácia')
print(metrics.accuracy_score(y_teste, pred))







#________________________________________________________________





