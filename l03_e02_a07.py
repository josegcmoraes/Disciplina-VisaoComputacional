# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
import matplotlib.pylab as plt
from sklearn import datasets, metrics
from sklearn import svm, naive_bayes, neighbors, model_selection
from matplotlib.colors import ListedColormap

# Carrega o conjunto de dados
'''
iris = datasets.load_iris()
iris.keys()
iris.data
iris.target
iris.target_names
print(iris.DESCR)
iris.feature_names
X = iris.data
y = iris.target
f = iris.feature_names
num_classes = len(np.unique(y))
num_feat = X.shape[1]
'''
iris=datasets.load_iris()
x=iris.data
y=iris.target
#x=x[:,[1,3]]
x=x[:,[1]]
#print(x)
y[y==1]=0
#y[y==3]=1

x_treino,x_teste,y_treino,y_teste=model_selection.train_test_split(x,y,test_size=0.25)

#print(y_treino)
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
    
 
class BayesSimples: 
    def dist_normal(self, h, media, desv_p): 
        """        
        """ 
        numerador = (1. / np.sqrt(2*np.pi * desv_p**2)) 
        denominador = np.exp(-((h - media)**2.) / (2 * desv_p**2))
        return numerador * denominador
    def fit(self, x_treino, y_treino): 
        """        
        """ 
        print('\nfit') 
        # Objetos das classes C0 e C1 
        classe_0 = x_treino[y_treino==0] 
        classe_1 = x_treino[y_treino==1]
        # Numero de objetos em cada classe 
        num_objs_c0 = classe_0.shape[0] 
        num_objs_c1 = classe_1.shape[0] 
        # Numero total de objetos (N) 
        num_objs = num_objs_c0 + num_objs_c1
        # Parametros das classes: média e desvio padrão 
        self.param_c0 = [np.mean(classe_0, 0), np.std(classe_0, 0)] 
        self.param_c1 = [np.mean(classe_1, 0), np.std(classe_1, 0)]
        # Caracteristica observada dos individuos (h) 
        self.h = np.linspace(np.min([classe_0.min(), classe_1.min()])-1., np.max([classe_0.max(), classe_1.max()])+1., 200)        
        # Funcao densidade condicional de h, sabendo que o objeto pertence a classe C0 ou C1: 
        # f(h|C0) e f(h|C1) 
        self.fh_c0 = self.dist_normal(self.h, self.param_c0[0], self.param_c0[1]) 
        self.fh_c1 = self.dist_normal(self.h, self.param_c1[0], self.param_c1[1])
        # A probabilidade de um individuo pertencer a classe C0 ou C1: 
        # P(C1) e P(C2) 
        self.p_c0 = float(num_objs_c0) / num_objs 
        self.p_c1 = float(num_objs_c1) / num_objs
        # ---- TESTE ---
        print('P(C0) = ', self.p_c0) 
        print('P(C1) = ', self.p_c1)
        # Funcao densidade de probabilidade condicional ponderada de h. 
        # P(C1)f(h|C1) e P(C2)f(h|C2): 
        self.p_c0_fh_c0 = self.p_c0 * self.fh_c0 
        self.p_c1_fh_c1 = self.p_c1 * self.fh_c1        
    def predict(self, x_teste): 
        """        Classifica um conjunto de objetos ainda não apresentados.        
        """ 
        print('\npredict')


BayesSimples().fit(x_treino, y_treino)
BayesSimples().predict(x_teste)

#BayesSimples.fit(x_treino,y_treino)
#BayesSimples.predict(x_teste)


#clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
#clf.fit(x_treino, y_treino)
#pred = clf.predict(x_teste)
print('\n Predição:') 
print(BayesSimples().predict(x_teste)) 
print('\nReal:') 
print(y_teste)
print('\nMatriz de confusão:') 
print(metrics.confusion_matrix(y_teste, BayesSimples.predict(x_teste)))
print('\nRelatório de classificação:') 
print(metrics.classification_report(y_teste, BayesSimples.predict(x_teste)))
print('acurácia')
print(metrics.accuracy_score(y_teste, BayesSimples.predict(x_teste)))



