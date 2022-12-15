# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:47:12 2022

@author: Osman VARIŞLI
"""

import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn import datasets



from sklearn.datasets import load_iris

#X= datalar , y=sonuçlar
X, y = load_iris(return_X_y=True) 

"""
#Şarap Dataseti
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)

#Göğüs kanseri dataseti
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

#diabet dataseti
from sklearn import preprocessing
with open("diabetes.csv", 'r') as x:
    diabet_data = list(csv.reader(x, delimiter=","))
diabet_data = np.array(diabet_data)
diabet_data = np.delete(diabet_data, 0, axis=0) #başlıkları siliyoruz
diabet_data = diabet_data.astype(np.float)

Xa=diabet_data[:,0:8] # illk 9 sutun verileri içermekte
Xa=preprocessing.scale(Xa)

ya=diabet_data[:,-1] #son sutun sonuçları göstermekte
X=Xa
y=ya
"""

#train_test_split ile verileeri eğitim ve sonuç olarak ikiye bölünüyor
#%33 lük veri test için kullanıldı.
#random state oranı rastgelelik numarası, eğitim ve test için aynı data setle çalışmak istersek random_state değerini aynı vermeliyiz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def bayes(X_train, y_train,X_test):
    
    def gausion_formulu(class_idx, x,varyanslar,ortalamalar):
        #gausion förmülü koda dönüştürülüyor.
        ortalama = ortalamalar[class_idx]
        varyanslar = varyanslar[class_idx]
        num = np.exp(- (x-ortalama)**2 / (2 * varyanslar))
        karekok = np.sqrt(2 * np.pi * varyanslar)
        return num / karekok 
    
    def Siniflandir(x,sinif,varyanslar,ortalamalar):
        toplamlar = []
        
        #her sınıf için bir döngü oluşturulup gausion fonksiyonuna gönderilip bunlar toplanılıyor
        #toplamı en büyük olan değerin sınıfı argmax ile seçiliyor.
        for i, c in enumerate(sinif):
            #gausian fonksiyonuna gönderip değerler toplanılıyor. 
            #veriler kaç boyutlu ise gausion fonksiyonu okadar boyulu veri dönderiyor bunlar toplanılıyr.
            toplam = np.sum(gausion_formulu(i, x,varyanslar,ortalamalar))
            
            toplamlar.append(toplam)
        
        #argmax ile en yüksek değere ait sınıf seçiliyor.
        return sinif[np.argmax(toplamlar)]
    
    satir, sutun = X.shape #verilerin boyutlarını çek
    sinif = np.unique(y_train) #sonuç değerlerini gurupla. iris çiçeği için 0,1,2 değerleri dönecektir.

    sinif_sayisi = len(sinif)#sonuçta çıkacak değerleri sayısnı göster. iris için 3 çıkacaktır.
    ortalamalar = np.zeros((sinif_sayisi, sutun), dtype=np.float64)#ortalama ve varyanlar için sıfır matrikslerini oluştur.
    varyanslar = np.zeros((sinif_sayisi, sutun), dtype=np.float64)
    for i, c in enumerate(sinif):
        Grupla = X[y==c]#sonuçları aynı olan datasetleri grupla

        if len(X[y==c])==1:Grupla=Grupla[0] #eğitim datasetin de yetersiz veri varsa varyansın sıfır çıkmasını engelle
        ortalamalar[i, :] = Grupla.mean(axis=0)#gruplanmış verilerde ortalamayı al
        varyanslar[i, :] = Grupla.var(axis=0)#gruplanmış verilerde varyansı al

    #yukarıda ki işlem ile her sonuç için ortalam ve varyans değerlerini aldık ve eğitimi bitirdiğimiş olduk
    #test verilerimizi tek tek Siniflandir fonksiyonuna gönderiyoruz.
    tahmin = [Siniflandir(x,sinif,varyanslar,ortalamalar) for x in X_test]
    return tahmin

tahmin =bayes(X_train, y_train,X_test) #fonksiyon çağır

print (accuracy_score(y_test, tahmin)) #başar oranını göster



