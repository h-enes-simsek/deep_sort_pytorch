import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(x.shape)
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


if __name__ == '__main__':
    #tek bir resim için model çalıştırılıp feature'lar kaydediliyor.

    import os
    from PIL import Image
    import torchvision.transforms.functional as TF
    
    dataPath = 'features' #kaydedilecek feature'lar için klasör
    if not os.path.isdir(dataPath):
        os.mkdir(dataPath)

    
    #random sayılar ya da bir resim yüklenmeli
    #inputs = torch.randn(64,3,128,64)
    
   
    #tek bir resim 
    image = Image.open('demo.jpg')
    inputs = TF.to_tensor(image)
    inputs.unsqueeze_(0) #3,x,y 'den 1,3,x,y 'ye
    #print(inputs.shape)
    
    
    #model bir kez çalıştırılıyor
    net = Net(reid=True)
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict, strict=False)
    net.eval() #dropout gibi optimizasyonları kapatmak için, net.train() ile geri açılıyor.
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    net.to(device)
    inputs = inputs.to(device)
    features = net(inputs).cpu()
    print(features.shape)
    print(features.norm())
    
    
    layers = [layer for layer in net.children()]
    print('Layers: {}'.format(len(layers)))
    print('Layers[0]: {}'.format(len(layers[0]))) #4
    print('Layers[1]: {}'.format(len(layers[1]))) #2
    print('Layers[2]: {}'.format(len(layers[2]))) #2
    print('Layers[3]: {}'.format(len(layers[3]))) #2
    print('Layers[4]: {}'.format(len(layers[4]))) #2
    #print('Layers[5]: {}'.format(len(layers[5])))#no length
    print('Layers[6]: {}'.format(len(layers[6]))) #4
    
      
       
    #net = Net()
    #inputs = torch.randn(5,3,128,64)
    y = net(inputs)
    #import ipdb; ipdb.set_trace()
    print(net) #alternatif bir model görme şekli torchsummarye göre
    from torchsummary import summary
    summary(net, (3,128,64))
    
    

