from numpy import loadtxt 
import numpy as np
import os
import matplotlib.pyplot as plt

def calc(case,seqFolder,trueOrFalse,saveOrShow): 
    mainFolder = "../../MOT16/train/" + seqFolder
    fileFolder = mainFolder + "/topScores"+str(case)+".txt"
    file = loadtxt(fileFolder, delimiter=",") #shape line_idx,list(7 eleman)
    file_size = file.shape[0]
       
    #k=1 veya k=0 olanları seç, yani doğru eşleşenler veya yanlış eşleşenler
    def filterType(param):
        if(trueOrFalse):
            if (param[2]==1):
                return True
            else:
                return False
        else:
            if (param[2]==0):
                return True
            else:
                return False

    file_filtered = list(filter(filterType, file))
    try:  
        top1Scores = np.array(file_filtered)[:,3] #sadece top1 skorları
    except:
        print("Hata: ",case,seqFolder,trueOrFalse,saveOrShow)
        return
    meanTop1 = np.mean(top1Scores) #seçili top1'lerin ortalaması
    varTop1 = np.var(top1Scores) #seçili top1'lerin varyansı

    #grafik adı
    if (case==1):
        label = seqFolder + " (Person, Visibility<.3)"
    elif (case==2):
        label = seqFolder + " (Person, .3<Visibility<.7)"
    elif (case==3):
        label = seqFolder + " (Person, All Visibility Scores)"
    elif (case==4):
        label = seqFolder + " (Car, All Visibility Scores)"
        
    #label adlerı
    if(trueOrFalse):
        xlabel = "Index of True Matches"
    else:
        xlabel = "Index of False Matches"
    ylabel = "Top1 Score"
    labelOfMean = "Top1 Mean ("+str.format('{0:.4f}', meanTop1)+")"
    labelOfVar = "Top1 Variance ("+str.format('{0:.4f}', varTop1)+")"

    fig = plt.figure()

    plt.title(label) #grafiğin adı
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 

    plt.hlines(meanTop1,0,top1Scores.shape[0],color="red",zorder=2, label=labelOfMean) #ortalamayı çiz
    plt.plot(top1Scores,"s",markersize=1,zorder=1) #verileri çizdir
    plt.plot([], [], ' ', label=labelOfVar) #varyansı legend'e eklemek için dummy plot
    plt.legend(loc="lower left") #legend

    if(saveOrShow):
        if not os.path.exists(mainFolder+"/../Top1Graphes"):
            os.makedirs(mainFolder+"/../Top1Graphes")
        plt.savefig(mainFolder+"/../Top1Graphes/"+seqFolder+"Case"+str(case)+str(trueOrFalse)+"Match")
    else:
        plt.show()
    
    plt.close(fig)

#hepsini tek seferde yapmak için loop
seqs_str = '''MOT16-02       
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13
                  ''' 
seqs = [seq.strip() for seq in seqs_str.split()]
for seq in seqs:
    calc(1,seq,True,1)
    calc(2,seq,True,1)
    calc(3,seq,True,1)
    calc(1,seq,False,1)
    calc(2,seq,False,1)
    calc(3,seq,False,1)

#eğer tek seferde hepsi çağrılmayacaksa parametreler 
#---------------PARAMETRELER----------------
case = 4 #1,2,3 veya 4 seçilmeli
seqFolder = "MOT16-13"
trueOrFalse = True #doğru veya yanlış eşleşmeler
saveOrShow = 1 #1 save, 2 show 
calc(case,seqFolder,trueOrFalse,saveOrShow)
#-------------------------------------------
    