import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            #[ortalama,ortalama,ortalama],[standart sapma,standart sapma,standart sapma] 
            #aşağıdaki kod her kanal için (rgb,en,boy) şunu hesaplıyor 
            #en kanalı için: en = (en - en_ort) / en_std
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])
        


    def _preprocess(self, im_crops):
        #im_crops, yolov3 tarafından üretilmiş bbox ile video frame'den kesilmiş parçalar.
        #kesilmiş parçaların boyutları farklı farklı
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            #im.astype(np.float32) numpy işlemi float 32 ye çevirmek için
            #pikseller 255'e bölünerek 0-1 arasına sıkıştırılıyor.
            #resize için interpolation belirtilmemiş, bilinear interpolation default kullanılıyor.
            #nihayetinde kesilmiş video parçası size=64,128 boyutuna getiriliyor.
            return cv2.resize(im.astype(np.float32)/255., size) 

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        #print(im_batch.shape) #detection_number,3,128,64
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            #print(features.shape) #detection_number,512
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

