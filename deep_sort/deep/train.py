import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root,"pytorch/train")
test_dir = os.path.join(root,"pytorch/val")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128,64),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    #[ortalama,ortalama,ortalama],[standart sapma,standart sapma,standart sapma] 
    #aşağıdaki kod her kanal için (rgb,en,boy) şunu hesaplıyor 
    #en kanalı için: en = (en - en_ort) / en_std
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64,shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64,shuffle=True
)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
#print("trainloader.dataset.classes: ",trainloader.dataset.classes) #market1501 için 751 class (class isimleri 2'den 1500'e aralıklarla) 
#print("testloader.dataset.classes: ",testloader.dataset.classes) #market1501 için 751 class (class isimleri 2'den 1500'e aralıklarla) 
#print("number of classes: ",num_classes) #market1501 için 751
#print(len(trainloader)) #191
#print(len(testloader)) #12


# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device) #.to pytorch tensor.to, tensorlerin özelliklerini değiştirmek için kullanılıyor. mesela from int32 to int64, from cude to cpu

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss() #https://pytorch.org/docs/0.3.1/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0. 
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        #print("idx: ",idx) #kaçıncı for döngüsü tespit etmek için
        #print("\ninputs: ",inputs.shape) #shape = 64,3,128,64 (ilk 64 batch size) (3,126,64 image) market 1501 için
        #ÇOK ÖNEMLİ, labellar klasör isimleri ile isimlendirilmiyor sırayla isimlendiriliyor. 0'dan 751'e
        #print("\nlabels: ",labels) #inputların class labeli, shape = 64 (batch size) market 1501 için
        # forward
        inputs,labels = inputs.to(device),labels.to(device) #.to pytorch tensor.to, tensorlerin özelliklerini değiştirmek için kullanılıyor. mesela from int32 to int64, from cude to cpu
        

        outputs = net(inputs) #64,751 = #batch_size,#class_number ?????????GÖZDEN GEÇİR????????
        #print("out: ",outputs[0,:].norm()) #SORUN: train esnasında çıkışın normu 1 değil?
        loss = criterion(outputs, labels) #outputlar probability değil. loss, softmax(outputs) işlemini gerçekleştiriyor. https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch
        print("loss: ",loss)
        
        #sadece cross entropinin doğru çalışıp çalışmadığını denetlemek için
        #batch için loss değerleri toplanıp ortalaması hesaplanıyor
        accumulate = 0;
        m = torch.nn.Softmax(dim=1)
        out = m(outputs)
        #print(out.shape) #64,751
        for i in range(64):
            #print("label: ",labels[i]," - ",out[i,labels[i]]) 
            #print(-1*torch.log(out[i,labels[i]] ))
            accumulate += -1*torch.log(out[i,labels[i]] )
        print(accumulate/64 == loss)
        
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        #loss.item() = loss (sadece tensorden float? a çevrilmiş.)
        training_loss += loss.item() #training_loss, aslında interval için ortalama loss'u hesaplamak için kullanılıyor
        #print("training_loss: ",loss.item())
        train_loss += loss.item() #epoch için asıl loss

        #outputs.max(dim=1) ile outputun içindeki her bir resim için (batch_size) en yüksek class ihtimaline sahip değeri seçiliyor
        #input_max, input_indexes = torch.max(input, dim=1) max olan index değerleri yani classlar [1] ile seçilioyr, [0] ise max değerlere eşit.
        #.eq ile seçili classlar ve olması gereken label classlar karşılaştırılıyor. sonuç true false olarak vectore yazılıyor.
        #.sum().item() ile truelar 1 falseler 0 olarak toplanıyor.
        correct += outputs.max(dim=1)[1].eq(labels).sum().item() 
        print("correct: ",outputs.max(dim=1)[1] )
        
        #correct/total karşılaştırması için labelların büyüklüğü hesaplanıyor. batch_size a eşit olmalı, 64
        total += labels.size(0)
        #print(labels.size(0))
        
        

        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s AverageLoss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    

    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch,40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()


if __name__ == '__main__':
    main()
