import torch
import torch.nn as nn
import torch.nn.functional as F

##Bu dosya model.py kodlarından türetilmiştir.

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751 ,reid=False): #num class sayısı market için 751
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1), #no stride, 1px padding
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        #print(x.shape)
        #show_image(x,'featureOrj')
        
        x = self.conv(x) 
        #print(x.shape) #1,64,64,32
        #show_feature(x,4,'cov1')
        
        x = self.layer1(x)
        #print(x.shape) #1,64,64,32
        #show_feature(x,5,'cov2')
        
        x = self.layer2(x)
        #print(x.shape) #1,128,32,16
        #show_feature(x,5,'cov3')
        
        x = self.layer3(x)
        #print(x.shape) #1,256,16,8
        #show_feature(x,5,'cov3')
        
        x = self.layer4(x)
        #print(x.shape) #1,521,8,4
        #show_feature(x,5,'conv5')
        
        x = self.avgpool(x)
        #print(x.shape) #1,512,1,1
        #show_feature(x,512,'avgpool')
        
        x = x.view(x.size(0),-1)
        #print(x.shape) #1,512
        
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True)) #L2 normalization by dividing its length.
            #print(x.shape) #1,512
            #show_last_feature(x)
            return x
        # classifier
        x = self.classifier(x)
        #print(x.shape)
        return x

#random tensorden img yazdırmak için, eğer girdi olarak resim değil rand sayı verildiyse    
def show_image(inp,saveName):
    inp = inp.view(3,128,64) #img id siliniyor
    import matplotlib.pyplot as plt
    import numpy as np
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    #plt.savefig(saveName+'.png')
    plt.show()
    plt.pause(.001)
    return

#son feature hariç feature map'leri kaydetmek için
def show_feature(inp,numberOfFeature,saveName):
    import matplotlib.pyplot as plt
    import numpy as np
    inp = inp.view(list(inp.size())[1:]) #img id siliniyor.
    inp = inp.detach().cpu()
    dataDir = dataPath+'/'+saveName
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    for i in range(numberOfFeature):
        feature = inp[i,:,:] #sıfırıncı feature
        #print(feature.size())

        plt.gray()
        plt.imshow(feature,extent=(0, feature.size(1), 0, feature.size(0))) #orjin noktası extent ile ayarlanıyor.
        plt.savefig(dataDir+'/'+str(i)+'.png')
    print(saveName," was saved into ",dataDir)
    return
    
def show_last_feature(feature):
    #print(feature)
    feature = torch.reshape(feature, (32, 16))
    feature = feature.detach().cpu()
    import matplotlib.pyplot as plt
    import numpy as np
    #print(feature.shape) #32,16
    
    dataDir = dataPath+'/'+'lastFeature'
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    plt.jet()
    plt.imshow(feature)
    plt.imshow(feature,extent=(0, feature.size(1), 0, feature.size(0))) #orjin noktası extent ile ayarlanıyor.
    plt.savefig(dataDir+'/feature.png')





#klasördeki resimler için network çalıştırılıp sonuçlar elde ediliyor.
#ancak değerlendirme yapılabilmesi için klasörün özel bir formatta olması gerekiyor.
#f<frame_num>id<tracking_id>.jpg
#her bir frame ile bir sonraki framedeki gt bbox ile kesilmiş resimler arasındaki benzerlik hesaplanıyor.

#yapılan testin sonuçları MOT-16 dosyasının ilgili videsonun içinde


import os
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import torchvision.transforms as transforms
import numpy as np
import re #regex
from numpy import loadtxt #gt.txt yi almak için
import time #debug


""" model.py dosyası için
dataPath = 'features' #kaydedilecek feature'lar için klasör
if not os.path.isdir(dataPath):
    os.mkdir(dataPath)
"""

#model yükleniyor
net = Net(reid=True)
assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval() #dropout gibi optimizasyonları kapatmak için, net.train() ile geri açılıyor.
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
net.to(device)
#----------------


#cropImgFolder = "../../MOT16GT/train/MOT16-09/cropsVisible07/"  #ek olarak Visibility>0.7 olanlar
#cropImgFolder = "../../MOT16GT/train/MOT16-09/crops/"   #Visibility filtresi yok, ignore ve id filtersi var
mainFolder = "../../MOT16/train/MOT16-02"
cropImgFolder = mainFolder + "/img1/" #üzerinde hiç işlem yapılmamış orjinal MOT videosunun frameleri
gtFolder = mainFolder +"/gt/gt.txt" #gt.txt dosyasının konumu



#1,2, 3 veya 4
#1: visibility<0.3
#2: visibility>0.3 ve visibility<0.7
#3: visibility>=0 (filtre yok)
#4: sadece araçları incele (visibility filtresi yok.)
filterName = 1


saveImage = 0 #1 ise resimleri kaydet, 0 ise kaydetme sadece sonuçları bul
#kaydedilecekse klasör konumu
#pathSave = "../../MOT16GT/train/MOT16-09/crops1/" #cropsların kaydedileceği klasör
#pathSave = "../../MOT16GT/train/MOT16-09/crops2/" #cropsların kaydedileceği klasör
pathSave = mainFolder + "/crops" +str(filterName)+"/" #cropsların kaydedileceği klasör

#videonun boyutlarını öğrenmek için ilk resim açılıyor
image = Image.open(cropImgFolder+"000001.jpg")
image = np.array(image)
width = image.shape[1] #video genişliği
height = image.shape[0] #video yüksekliği



#gt verileri
#-------------------------------------------------------------------------
gt = loadtxt(gtFolder, delimiter=",") #shape line_idx,list(9 eleman)
gt_size = gt.shape[0]   

#-----------------------------
#frame'e göre sıralama
def sortwithFrame(elem):
    return elem[0]

# sort list with key
gt_sorted = sorted(gt,key=sortwithFrame)

"""
for i in range (3): #limit gt_size
  print(gt_sorted[i])
"""



#-----------------------------
# object_type=1 olmayanları sil, 
def filterType(param):
    if (param[7]==1):
        return True
    else:
        return False

gt_filtered = list(filter(filterType, gt_sorted))
"""
for i in range (3): #limit gt_size
  print(gt_filtered[i])
"""



#-------------------------------
#not_ignored=0  olanları sil
def filterIgnore(param):
    if (param[6]==1):
        return True
    else:
        return False

gt_filtered2 = list(filter(filterIgnore, gt_filtered))
"""
for i in range (30): #limit gt_size
  print(gt_filtered2[i])
"""
# gt_filtered3 = np.array(gt_filtered3) #from python array to np array
# print(gt_filtered2.shape) #5257,9 for MOT-9



#-------------------------------
#visiblity filtresi
def filterVisible(param):
    if filterName==1:
        if (param[8]<0.3):
            return True
        else:
            return False
    if filterName==2:
        if (param[8]>0.3 and param[8]<0.7):
            return True
        else:
            return False
    if filterName==3:
        if (param[8]>=0):
            return True
        else:
            return False

gt_filtered3 = list(filter(filterVisible, gt_filtered2))



"""
for i in range (30): #limit gt_size
  print(gt_filtered3[i])
"""
gt_filtered3 = np.array(gt_filtered3) #from python array to np array
#print(gt_filtered3.shape) #2624,9 for MOT-9
#-------------------------------------------------------------------------




#araçlar üzerinde test yapılamk isteniyorsa filterName=4 olmalı.
#burda gt_filtered3, gt_sorted'tan sonra tekrar hesaplanıyor. Ancak bunun yerine kodlar daha düzenli hale getirilebilir.
if filterName==4:
    def filterCar(param):
        if (param[7]==3):
            return True
        else:
            return False
    gt_filtered3 = list(filter(filterCar, gt_sorted))
    gt_filtered3 = np.array(gt_filtered3) #from python array to np array


norm = transforms.Compose([
        transforms.ToTensor(),
        #[ortalama,ortalama,ortalama],[standart sapma,standart sapma,standart sapma] 
        #aşağıdaki kod her kanal için (rgb,en,boy) şunu hesaplıyor 
        #en kanalı için: en = (en - en_ort) / en_std
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])


"""
#dosya isimleri
from os import walk
for (dirpath, dirnames, filenames) in walk(cropImgFolder):
    break

#düzgün sıralamak için, human sorting
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

filenames.sort(key=natural_keys)
#print(filenames) dosya içimlerini içeren tüm liste
"""


#gerekli fonksiyonlar
def _xywh_to_xyxy(x,y,w,h):
    global width,height
    x1 = max(int(x),0)
    x2 = min(int(x+w),width-1)
    y1 = max(int(y),0)
    y2 = min(int(y+h),height-1)
    return x1,y1,x2,y2
        
def _get_roi(x,y,w,h, ori_img): #tlwh formatındaki bbox ile frame kesiliyor.
    im_crops = []
    x1,y1,x2,y2 = _xywh_to_xyxy(x,y,w,h) #x1,y1,en,boy 'dan x1,y1,x2,y2 'ye
    #print("x1x2y1y2",x1,x2,y1,y2)
    #print("xywh ",x,y,w,h)
    im = ori_img[y1:y2,x1:x2] #tüm frame'den bbox ile seçilmiş kısmı alınıyor, rgb kanalı default seçiliyor. #boy,en,3
    im_crops.append(im)
    return im_crops

resultTXT = torch.tensor([]).float() #top1,2 sonuçlarını frame frame yazdırmak için hazırlanan array

def write_results(filename, results):
    #is correct yanlış eşleşmelerde k yi gösteriyor. (topk içindeki)
    save_format = '{frame},{id},{is_correct},{top1},{top2},{visib_curr},{visib_next}\n'
    with open(filename, 'w') as f:
        for frame, id, is_correct, top1, top2, visib_curr, visib_next  in results:
            #if track_id < 0: #TODO: bu yöntem is_correct sıfırken kullnaılacak
            #    continue
            line = save_format.format(frame=frame, id=id, is_correct=is_correct, top1=top1, top2=top2, visib_curr=visib_curr, visib_next=visib_next)
            f.write(line)

sumMaxMatches = 0
sumMatchNumber = 0
topkcorrect = 0

sumMaxMatches2 = 0
sumMatchNumber2 = 0
topkcorrect2 = 0

def calc_similarity(f,arr_curr,arr_next):
    #start = time.process_time() #zamanı ölçmek için
    
    if(len(arr_curr)==0 or len(arr_next)==0): #eğer arraylerden biri boşsa dur
        print("one of frame has no object, frame: ",f)
        return
    
    print("frame:", f)
    
    arr_curr = np.array(arr_curr).astype(float)
    arr_next = np.array(arr_next).astype(float)
     
    visib_curr = [] #visibilty skorlarını almak için
    visib_next = [] #visibilty skorlarını almak için
    
    global resultTXT
    global sumMaxMatches, sumMatchNumber, topkcorrect
    global sumMaxMatches2, sumMatchNumber2, topkcorrect2
    
    features_curr = torch.tensor([]).float()
    features_next = torch.tensor([]).float()
    arr_same = list(set(arr_curr[:,0]).intersection(arr_next[:,0])) #her iki frame'de de bulunan kişiler (idleri kontrol ediliyor)
    #print(arr_curr)    
    #print(arr_next)
    #print(f,' ',arr_same)
    
    path_curr = cropImgFolder+format(int(f), '06d')+".jpg" #şu anki frame
    path_next = cropImgFolder+format(int(f+1), '06d')+".jpg" #bir sonraki frame
    
    imageOrj = Image.open(path_curr)
    image = np.array(imageOrj)
    
    imageOrj2 = Image.open(path_next)
    image2 = np.array(imageOrj2)
    
    #print(arr_same)
    with torch.no_grad():
        #print(arr_same)
        for idx,val in enumerate(arr_same):
        
            
            id,x,y,w,h,vis=arr_curr[arr_curr[:,0]==val][0,:] #her iki framede de olan(arr_same de olan) idler için bbox bilgisi
            id = int(id)
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            #print(id,x,y,w,h)
            crop_img = _get_roi(x,y,w,h, image)
            crop_img = crop_img[0]
            
            _,xN,yN,wN,hN,visN=arr_next[arr_next[:,0]==val][0,:] #her iki framede de olan(arr_same de olan) idler için bbox bilgisi
            xN = int(xN)
            yN = int(yN)
            wN = int(wN)
            hN = int(hN)
            crop_img2 = _get_roi(xN,yN,wN,hN, image2)
            crop_img2 = crop_img2[0]
            
            if (saveImage==1):  #1 kaydet, 0 kaydetme
                if not os.path.isfile(pathSave+"/f"+str(int(f))+"id"+str(int(id))+".jpg"):
                    try:
                        crop_imgShow = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        status = cv2.imwrite(pathSave+"/f"+str(int(f))+"id"+str(int(id))+".jpg",crop_imgShow)
                    except:
                        print(f,id,x,y,w,h)
                        print("Yazdırılırken hata oluştu.")
            
            #resmi göster
            #crop_imgShow = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) 
            #cv2.imshow('image',crop_imgShow)
            #cv2.waitKey()
            
            
            inputs = cv2.resize(crop_img.astype(np.float32)/255., (64, 128)) 
            inputs = norm(inputs)
            inputs = inputs.unsqueeze_(0) #3,x,y 'den 1,3,x,y 'ye
            inputs = inputs.float()
            inputs = inputs.to(device)
            features1 = net(inputs).cpu()
            features_curr = torch.cat((features_curr, features1), dim=0)
            
            
            
            inputs2 = cv2.resize(crop_img2.astype(np.float32)/255., (64, 128)) 
            inputs2 = norm(inputs2)
            inputs2 = inputs2.unsqueeze_(0) #3,x,y 'den 1,3,x,y 'ye
            inputs2 = inputs2.float()
            inputs2 = inputs2.to(device)
            features2 = net(inputs2).cpu()
            features_next = torch.cat((features_next, features2), dim=0)
            
            #visibiltyleri bir arraya ekle
            visib_curr.append(vis)
            visib_next.append(visN)
            
            
    imageOrj.close()
    imageOrj2.close()
    
    if(len(arr_same)==0): #eğer hiç ortak obje yoksa dur
        print("no joint object, frame: ",f)
        return
        
    #TOPk verilerinin bulunduğu kısım -----------------------------------------------------
    
    arr_same = torch.FloatTensor(arr_same)
    scores = features_curr.mm(features_next.t()) #curr,next image benzerlik skorları
    res = scores.topk(arr_same.size(0), dim=1) #detaylı bilgi için >> evaluate.py, arr_same'in boyutu kadar top skoru bulunuyor.
    
    
    #TOP1 
    sumMatchNumber = sumMatchNumber + arr_same.size(0)
    

    selector = arr_same[res[1][:,0]].eq(arr_same)
    newtopkcorrect = selector.sum().item()
    topkcorrect = topkcorrect + newtopkcorrect
    
    #print(arr_same[~(arr_same[res].eq(arr_same))])
    
    maxMatches = res[0][:,0] #en yüksek eşleşmeler için skor değerleri
    maxMatches_sum_local = torch.sum(maxMatches)

    sumMaxMatches = sumMaxMatches + maxMatches_sum_local
    
    """
    if (int(newtopkcorrect)!=int(arr_same.size(0))): #her yanlışta top1'i yazdır
        #print("Acc top1:{:.3f}".format(topkcorrect/sumMatchNumber)) 
        #print("Avg top1: ",sumMaxMatches/sumMatchNumber)
        print("wrong match in frame: ",f)
        #print("The wrong match :       ",arr_same[res])
        #print("The match it should be :",arr_same)
        #print(scores)
    """
    
    
    #TOP2
    flag_top2_exist = 1
    try: #benzerlik matrisi 1x1 olduğunda top2 hata veriyor
        sumMatchNumber2 = sumMatchNumber2 + arr_same.size(0)
        
        selector2 = arr_same[res[1][:,1]].eq(arr_same)
        newtopkcorrect2 = selector2.sum().item()
        topkcorrect2 = topkcorrect2 + newtopkcorrect2

        maxMatches2 = res[0][:,1] #en yüksek ikinci eşleşmeler için skor değerleri
        maxMatches2_sum_local = torch.sum(maxMatches2)

        sumMaxMatches2 = sumMaxMatches2 + maxMatches2_sum_local
    except:
        flag_top2_exist = 0
        print("similarty matrix 1x1, frame: ",f)
    
    
    #Çok fazla hatalı çıktı olduğu için yazdırmanın bir anlamı yok
    
    #if (int(newtopkcorrect)!=int(arr_same.size(0))): 
    #    #print("Acc top1:{:.3f}".format(topkcorrect/sumMatchNumber)) 
    #    #print("Avg top1: ",sumMaxMatches/sumMatchNumber)
    #    print("frame: ",f)
    #    print("The wrong match :       ",arr_same[res])
    #    print("The match it should be :",arr_same)
    #    print(scores)
    
    #-------------------------------------------------------------------------------------------
    
    #boolean top1 selector'u 1/0 arraye çevir
    is_correct = list(map(int, selector))
    
    if flag_top2_exist == 0:
        #top2 yerine -1 yazılıyor.
        top2 = [-1] * arr_same.size(0)
    else:
        #top2 = maxMatches2[selector] #doğru eşleşenlerin top2 sonuçları
        top2 = maxMatches2 #tümtop2 sonuçları
    
    frame = [f] * arr_same.size(0)
    id = arr_same
    #top1 = maxMatches[selector] #doğru eşleşmeler için top1 sonuçları
    top1 = maxMatches #tümtop1 sonuçları
    
    

    #print("sonuc: " ,resultTXT)    
    """    
    print(scores)
    print(maxMatches2)
    print(maxMatches)
    print(arr_same)
    print(selector)
    print(selector2)
    print("top1: ",maxMatches[selector]) #top1 doğru ise top1
    print("top2: ",maxMatches2[selector]) #top1 doğru ise top2
    """
    
    
    
    #sadece diagonal elemenlentleri çarpan kod (top1 sonuç bulmak için alternatif ama sadece acc:100 iken işe yarar)
    """
    l = len(arr_same)
    for i in range (l ):
        res = features_curr[i] * features_next[i]
        sumMaxMatches = sumMaxMatches + torch.sum(res)
    
    sumMatchNumber = sumMatchNumber + l 
    #print(sumMaxMatches,' - ',sumMatchNumber)
    print('Avg= ',sumMaxMatches/sumMatchNumber) #latest avg value
    

    return
    """
    for i in range (len(arr_same)):
        if selector[i]==True or len(arr_same)==1 or len(arr_same)==0:
            continue
        k = sorted(scores[i,:], key=float, reverse=True).index(scores[i,i]) #top k deki k ifadesi
        print(scores)
        is_correct[i] = k +1
        top2[i] = scores[i,i] #eşleşmesi gereken imagelerin benzerlik skoru.
    
    print(frame,id,is_correct,top1,top2,visib_curr,visib_next)
    
    #ilk terimdeki frame, f ve f+1. frameler için eşleşmeleri ifade ediyor.
    a = np.stack((frame,id,is_correct,top1,top2,visib_curr,visib_next),axis=-1)
    resultTXT  = torch.cat( (resultTXT,torch.from_numpy(a) ) , 0)
    
    # print(time.process_time() - start) #zamanı ölçmek için
    


temp_current_frame = 1
temp_next_frame = 2
temp_current_tracking = []
temp_next_tracking = []


#yaptığım bir hatayı düzeltmek için
###### YAZDIRIRKEN ESKİ DOSYALARA OVERWRİTE ETME ###########
"""
fileFolderTopScores = "deneme.txt"
file = loadtxt(fileFolderTopScores, delimiter=",") #shape line_idx,list(7 eleman)
f_topScores = file[:,0]
k_topScores = file[:,2]
index = [j for j,k in enumerate(k_topScores) if k!=1]
file_only_k = file[index]
frames_only_k = file_only_k[:,0]
ids_only_k = file_only_k[:,1]
startFlag = True
secondFlag = False
"""
#-----------------------------------

for i,name in enumerate(gt_filtered3):   

    frame,id,x,y,w,h,not_ignored,object_type,visibility=gt_filtered3[i] #gt.txt den gerekli verileri al
    frame = int(frame)
    id = int(id)
    #print(frame)
    #gt'den filtreli isimleri almanın alternatifi: klasördeki bütün veriler uygun şekilde tespit ediliyor. (eğer klasör daha önce oluşturulduysa)
    #frame = int(re.search('f(.*)i', name).group(1)) 
    #id = int(re.search('d(.*).jpg', name).group(1)) 
    
    #---------------------------------------------
    """
    #index2 = [j for j,k in enumerate(frames_only_k) if k==frame]
    if not(frame in frames_only_k):
        continue
    
    print(frame," ", id)

    if(startFlag):
        temp_current_frame = frame
        startFlag = False
        secondFlag = True
        flagFirstVal = frame
    elif(secondFlag and frame!=flagFirstVal):
        temp_next_frame = frame
        secondFlag = False
        
    """
    #----------------------------------------
        

    idAndBbox = id,x,y,w,h,visibility
    idAndBbox = list(idAndBbox)
    
    #ardışık iki framedeki cropları karşılaştırmak için framelerde hangi cropların bulunduğu bilgisine ihtiyaç var.
    #iki array, şimdiki ve bir sonraki framedeki idleri ve bboxları tutuyor.
    if(temp_current_frame==frame):
        temp_current_tracking.append(idAndBbox)
    elif(temp_next_frame==frame):
        temp_next_frame = frame
        temp_next_tracking.append(idAndBbox)
        
        if(len(gt_filtered3)==i+1): #son elemanı tespit etmek için
            calc_similarity(frame-1,temp_current_tracking,temp_next_tracking)
            #print(temp_next_tracking)
    else:
        calc_similarity(frame-2,temp_current_tracking,temp_next_tracking)
        #print(temp_next_tracking) #her yeni array de yazdır
        temp_current_tracking.clear()
        temp_current_tracking = temp_next_tracking.copy()
        temp_next_tracking.clear()
        temp_next_tracking.append(idAndBbox)
        temp_current_frame = temp_current_frame+1
        temp_next_frame = frame
    
  
write_results(mainFolder+"/topScores"+str(filterName)+".txt",resultTXT) #sonuçları bir txt dosyasına yazdır  
    
#print("Acc top1:{:.3f}".format(topkcorrect/sumMatchNumber))
#print("Avg top1: ",sumMaxMatches/sumMatchNumber)

print("Acc top2:{:.3f}".format(topkcorrect2/sumMatchNumber2))
print("Avg top2: ",sumMaxMatches2/sumMatchNumber2)
print("toplam eşleşme: ",sumMatchNumber)
    



