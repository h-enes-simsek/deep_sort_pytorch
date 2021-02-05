import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
#orj: parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--data-dir",default='data/pytorch',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir

query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=64, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=64, shuffle=False
)
#print(len(queryloader.dataset.classes)) #750 class (train_set_class=751'e eşit olmak zorunda değil. market 1501, 750+751=1501 adet class içeriyor.)
#print(len(galleryloader.dataset.classes)) #752 class (fazla olan class 0 ve -1, bunlarda istenmeyen resimler var)
#print(len(queryloader)) #53
#print(len(galleryloader)) #509

# net definition
net = Net(reid=True)
assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.t7')
# orj: checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
#ancak cuda'sız çalışması için yeni bir parametre eklenmeli
checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval() #dropout gibi optimizasyonları kapatmak için, net.train() ile geri açılıyor.
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx,(inputs,labels) in enumerate(queryloader):
        #print(labels) #ÇOK ÖNEMLİ, labellar klasör isimlerine göre isimlendirilmiyor. index numarasına göre isimlendiriliyor.
        inputs = inputs.to(device)
        features = net(inputs).cpu() #64,512 = batch_size, ZQPei'nin orjinal olmayan modelindeki feature büyüklüğü
        #print("features: ",features.shape)
        query_features = torch.cat((query_features, features), dim=0) 
        #print("query_features: ",query_features.shape) # 3368,512 = market 1501 deki query_image, feature
        query_labels = torch.cat((query_labels, labels))
        #print("query_labels: ",query_labels.shape) #3367

    for idx,(inputs,labels) in enumerate(galleryloader):
        #print(labels) #ÇOK ÖNEMLİ, labellar klasör isimlerine göre isimlendirilmiyor. index numarasına göre isimlendiriliyor.
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        #print("gallery_features: ",gallery_features.shape) # 19732,512 = market 1501 deki test image, feature
        gallery_labels = torch.cat((gallery_labels, labels))
        #print("gallery_labels: ",gallery_labels.shape) # 19732

#MARKET 1501 gallery(test) verisetinde -1 ve 0 numaralı classları olduğu için sonuçlardan çıkarılıyor.
#Aslında çıkarılmıyor, tüm classların labelı iki azaltılıyor. query içinde -1 ve 0 olmadığı için, gallery ve query labelları eşleşmiş oluyor.
gallery_labels -= 2  #query_label = 0,0,0,1,1,.... iken gallery_label(işlem sonrası) = -2,-2,-2,-1,-1,0,0,0,1,1,1 ...

# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
}
torch.save(features,"features.pth")