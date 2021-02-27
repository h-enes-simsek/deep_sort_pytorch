#bu dosyayı topk skorları yazdırırken yaptığım bir hatayı düzeltmek için oluşturdum
#evaluateCrops.py top1, top2 skorlarını doğru yazdırdı eğer top1 skoru doğru eşleşmeyi veriyorsa
#ancak yanlış eşleşme durumunda top2 yerine topk yazılacak idi. Ben yerine sıfır yazdırdım.
#bu hatayı düzeltmek için üretilmiş topScores.txt dosyları içindeki k=1 den farklı durumlar tekrar değerlendirilecek ve 
#yerlerine doğru değerler yazılacak.

from numpy import loadtxt 
import numpy as np
import os
import matplotlib.pyplot as plt

#mainFolder = "../../MOT16/train/" + seqFolder
#fileFolder = mainFolder + "/topScores"+str(case)+".txt"
fileFolder = "deneme.txt"
file = loadtxt(fileFolder, delimiter=",") #shape line_idx,list(7 eleman)

#for i,name in enumerate(file):   
f=file[:,2] #txt den gerekli verileri al
index = [j for j,k in enumerate(f) if k!=1]

frames_k = file[index][:,0]
print(len(index))
    
"""
if (k==0 or k!=1):
    print("ESKI ",f," ",id," ",top1," ",top1)
    
    path_curr = cropImgFolder+format(int(f), '06d')+".jpg" #şu anki frame
    path_next = cropImgFolder+format(int(f+1), '06d')+".jpg" #bir sonraki frame
    
    imageOrj = Image.open(path_curr)
    image = np.array(imageOrj)
    
    imageOrj2 = Image.open(path_next)
    image2 = np.array(imageOrj2)
"""
        
    
    