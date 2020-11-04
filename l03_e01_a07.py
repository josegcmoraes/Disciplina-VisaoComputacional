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
x=x[:,[1,3]]
y[y==1]=0
y[y==3]=1
print('___x__')
print(x)
print('__y___')
print(y)
print('_____')
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
    
 
class KnnSimples:     
    def fit(self, x_treino, y_treino):
        """        
        """ 
        print('\nfit') 
        self.x_treino = x_treino 
        self.y_treino = y_treino
            
    def predict(self, x_teste): 
        """        Classifica um conjunto de objetos ainda não apresentados.   
        """ 
        print('\npredict')
        self.x_teste = x_teste
        
        # Calcula as distâncias entre os objetos do conjunto de testes e os objetos do conjunto de treino 
        dist_mat = spatial.distance.cdist(x_teste, self.x_treino, 'euclidean') 
        print('\nMatriz de distâncias: ')
        print(dist_mat.shape) 
        print(dist_mat)
        
        # Obtém os indices dos objetos do conjunto de treino mais próximos de cada o bjeto do conjunto de testes. 
        min_dist_i =dist_mat.argmin(axis=1) 
        print('\nÍndices dos vizinhos mais próximos: ')
        print(min_dist_i.shape) 
        print(min_dist_i)
        
        # Obtém a classe dos objetos no conjunto de treino mais próximos de cada obj eto do conjunto de testes. 
        y_pred = self.y_treino[min_dist_i] 
        print('\nClasses dos vizinhos mais próximos: ') 
        print(y_pred.shape) 
        print(y_pred)
        
        return y_pred





clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(x_treino, y_treino)
pred = clf.predict(x_teste)

#KnnSimples.fit(x_treino, y_treino)
#pred = KnnSimples.predict(x_teste)
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


map_a = ListedColormap(['r', 'g', 'b']) 
map_b = ListedColormap(['lightcoral', 'lightgreen', 'skyblue'])
h = .02 # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each 
# point in the mesh [x_min, x_max]x[y_min, y_max]. 
x_min, x_max = x_treino[:, 0].min() - 1, x_treino[:, 0].max() + 1 
y_min, y_max = x_treino[:, 1].min() - 1, x_treino[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot 
Z = Z.reshape(xx.shape)
plt.figure(figsize=(7,7)) 
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light) 
plt.pcolormesh(xx, yy, Z, cmap=map_b)

# Plot the training points 
plt.scatter(x_treino[:, 0], x_treino[:, 1], c=y_treino, cmap=map_a, edgecolor='k', s =80) 
plt.xlim(xx.min(), xx.max()) 
plt.ylim(yy.min(), yy.max()) 
plt.title("Classificador K-NN")
# Plot also the training points 
#plt.scatter(x_teste[:, 0], x_teste[:, 1], c=y_teste, cmap=map_a, edgecolor='k', mark er='^' s=80) 
plt.scatter(x_teste[:, 0], x_teste[:, 1], c=y_teste, cmap=map_a, edgecolor='k',marker='^', s=80) 
plt.title("Classificador K-NN")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
   
    
    