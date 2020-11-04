# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
import matplotlib.pylab as plt
from sklearn import datasets, metrics
from sklearn import svm, naive_bayes, neighbors, model_selection
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi 
from skimage import data,img_as_float,filters,measure,color,morphology
import imageio
import os

diretorio=os.listdir('data/mpeg7_2/')
print(diretorio)
print("________")
im=[]
imdata=[]
imtarget=[]
i=0
tg=0
F=np.array([])
for imagem in diretorio:
    print(imagem)
    img=imageio.imread('data/mpeg7_2/'+imagem)
    img=img_as_float(img)
    im_mf = ndi.filters.median_filter(img, size=3) 
    val1 = 0.9 
    im_bw = im_mf <= val1 
    se = np.ones([3,3]) 
    im_ml = morphology.binary_opening(im_bw, se) 
    im_ml = morphology.binary_closing(im_ml, se) 
    im_ml = morphology.remove_small_objects(im_ml, 40)
    im_r = measure.label(im_ml, background=0) 
    F = np.array([]) 
    print (type(F), F.shape)
    props = measure.regionprops(im_r, img) 
    
    plt.figure()
    plt.imshow(img)
    plt.show()
    print("________")
    im.append(img)
    for imgmini in range(0,im_r.max()): 
        F_i = np.array([props[imgmini].area,props[imgmini].eccentricity,props[imgmini].extent,props[imgmini].solidity])
        
        i=i+1;
        #F=np.array([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity])
        imdata.append(F_i)
        #imdata.append(np.array([props[0].area,props[0].eccentricity,props[0].extent,props[0].solidity]))
        print(F_i)
        imtarget.append(tg)
    tg=tg+1
    



print("\n\ndata")
print(imdata)
print("\n\ntarget")
print(imtarget)
     
#print(im)
#print(imagens)


x=imdata
y=imtarget
'''
x=x[:,[1,3]]
y[y==1]=0
y[y==3]=1
'''
'''
x_treino,x_teste,y_treino,y_teste=model_selection.train_test_split(x,y,test_size=0.25)
print("\n\n x_treino")
print(x_treino)
skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
i = 0
for train_index, test_index in skf.split(x, y):
    print('Fold ' + str(i))
    print(train_index)
    x_treino, x_teste = x[train_index], x[test_index]
    y_treino, y_teste = y[train_index], y[test_index]
    
    print('Índices:')
    print('Treino: ' + str(train_index))
    print('Teste: ' + str(test_index))
    print('\nRótulos:')
    print('Treino: ' + str(y_treino))
    print('Teste: ' + str(y_teste))
    
    print(x_treino)
    #print(x_teste)
    i=i+1
    print()


'''
x_treino, x_teste, y_treino, y_teste = model_selection.train_test_split(imdata, imtarget, test_size=0.25, random_state =42)
#x_treino = x_treino[:,:2] 
#x_teste = x_teste[:,:2] 

'''
from matplotlib import mlab 
mlab_pca = mlab.PCA(F) 
for i in range(x_treino):
    plt.scatter(mlab_pca.Y[T==i,0], mlab_pca.Y[T==i,1], marker=lm[i], color=lc[i])      

'''




'''
# Informacoes sobre o conjunto de dados. 
# Numero total de objetos (N) 
# Numero de caracteristicas (M) 
N, M = F.shape 
# Numero de classes (K) 
K = np.unique(T).shape[0] 
 
# Calcula o vetor medio 2D. 
F_mean = np.mean(F, 0) 
print ('Vetor medio: ') 
print (F_mean)
 
# Calcula a matriz de covariancia  
F_cov_mat = np.cov(F[:,0], F[:,1]) 
print ('Matriz de covariancia: ') 
print (F_cov_mat)
 
# Calcula o coeficiente de correlação de Pearson. 
F_corr = np.corrcoef(F[:,0], F[:,1]) 
print ('Coeficiente de correlacao: ')
print (F_corr)

# Calcular os autovetores e autovalores. 
auto_val, auto_vet = np.linalg.eig(F_cov_mat) 
print ('\nAutovalor: ') 
print (auto_val) 
print ('\nAutovetor: ') 
print (auto_vet) 
 
# Calcula a matriz de caracteristicas transformada (F_novo). 
F_novo = np.dot(auto_vet.transpose(), F.transpose()).transpose() 
print ('\nNova matriz de caracteristicas: ') 
print (F_novo) 
 
# Ordena as novas caracteristicas (componentes)  
# de acordo com os auto valores. 
ordem = np.argsort(auto_val,)[::-1] 
print ('\nComponentes ordenados: ') 
print (ordem) 
 
F_novo = F_novo[:,ordem] 
 
# Coeficiente de correlação de Pearson. 
# Apos transformacao. 
F_novo_corr = np.corrcoef(F_novo.transpose()) 
print ('\nCoeficiente de correlacao (após transformacao): ')
print (F_novo_corr) 
'''




N=i;
M=len(x[0])
#k=tg
k=np.unique(y).shape[0]
#N,M=x.shape

print(N)
print(M)
print(k)
x_mean=np.mean(x,0)
print('x_mean')
print(x_mean)

x_cov_mat=np.cov(x[0][0],x[0][1])
print(x_cov_mat)



clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(x_treino, y_treino)





clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(x_treino, y_treino) 
pred = clf.predict(x_teste)
print('\n\nClassificador K-NN..................................')
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
print('\n\nClassificador Bayes.................................')
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
print('\n\nClassificador SVM.................................')
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





# Plota a imagem
plt.figure()
#for i in im:
    #plt.imshow(im[i])
#plt.imshow(img)
plt.show()