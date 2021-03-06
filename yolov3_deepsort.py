import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from numpy import loadtxt #gt.txt yi almak için

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
            
        #eğer gt'den veriler okunacaksa    
        if self.args.gt:
            gtFolder = self.video_path + "/../gt/gt.txt"
            gt = loadtxt(gtFolder, delimiter=",")
            
            def sortwithFrame(elem):
                return elem[0]

            # sort list with key
            gt_sorted = sorted(gt,key=sortwithFrame)
            
            #-----------------------------
            # object_type=1 olmayanları sil, 
            def filterType(param):
                if (param[7]==1):
                    return True
                else:
                    return False

            gt_filtered = list(filter(filterType, gt_sorted))

            #-------------------------------
            #not_ignored=0  olanları sil
            def filterIgnore(param):
                if (param[6]==1):
                    return True
                else:
                    return False

            gt_filtered2 = list(filter(filterIgnore, gt_filtered))
            
            self.gt = np.array(gt_filtered2)
            

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
            
    #deep_sort içindeki fonksiyon doğru çalışmadığı için düzenleyip buraya fonksiyon olarak yazdım.
    #input: frame görüntüsü, xywh formatında bbox matrisi (shape=#ofDetections,4)
    #output: xywh formatında matrisin xyxy formatında matris karşılığı
    def my_xywh_to_xyxy(self,ori_img, bbox_xywh): 
        x,y,w,h = bbox_xywh[:,0],bbox_xywh[:,1],bbox_xywh[:,2],bbox_xywh[:,3]
        x = x.reshape((x.size,1))
        y = y.reshape((y.size,1))
        w = w.reshape((w.size,1))
        h = h.reshape((h.size,1))
        #ekranın boyutu alınıyor
        height, width = ori_img.shape[:2]
        x1 = np.maximum(np.int_(x-w/2),0)
        x2 = np.minimum(np.int_(x+w/2),width-1)
        y1 = np.maximum(np.int_(y-h/2),0)
        y2 = np.minimum(np.int_(y+h/2),height-1)
        arr = np.concatenate((x1,y1,x2,y2),axis=1)
        return arr
        
    def my_tlwh_to_xywh(self,ori_img, bbox_tlwh): 
        x,y,w,h = bbox_tlwh[:,0],bbox_tlwh[:,1],bbox_tlwh[:,2],bbox_tlwh[:,3]
        x = x.reshape((x.size,1))
        y = y.reshape((y.size,1))
        w = w.reshape((w.size,1))
        h = h.reshape((h.size,1))
        #ekranın boyutu alınıyor
        height, width = ori_img.shape[:2]
        x1 = np.minimum(np.int_(x+w/2),width-1)
        y1 = np.minimum(np.int_(y+h/2),height-1)
        arr = np.concatenate((x1,y1,w,h),axis=1)
        return arr
        
    #topleft(xy)wh >> xyxy dönüştürücü
    #gt içinde veriler tlxy şeklinde verilmiş. yolo verilerini xywh olarak üretiyor. (xy orta nokta)
    def my_tlwh_to_xyxy(self,ori_img, bbox_tlwh): 
        x,y,w,h = bbox_tlwh[:,0],bbox_tlwh[:,1],bbox_tlwh[:,2],bbox_tlwh[:,3]
        x = x.reshape((x.size,1))
        y = y.reshape((y.size,1))
        w = w.reshape((w.size,1))
        h = h.reshape((h.size,1))
        #ekranın boyutu alınıyor
        height, width = ori_img.shape[:2]
        x1 = np.maximum(np.int_(x),0)
        x2 = np.minimum(np.int_(x+w),width-1)
        y1 = np.maximum(np.int_(y),0)
        y2 = np.minimum(np.int_(y+h),height-1)
        arr = np.concatenate((x1,y1,x2,y2),axis=1)
        return arr    

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            
            #print(im.shape) #video_boyu,video_eni,3

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im) #bbox_xywh, confidence, labels 

            #gt'leri gt'den okuyarak yolo yerine veren kısım
            if (self.args.gt): #py çalıştırılırken --gt yazıldıysa
                if(idx_frame == 1 or idx_frame == 2 or idx_frame == 3): #üç frame boyunca gt verileri yolo yerine veriliyor
                    gt_curr_frame = self.gt[self.gt[:,0]==idx_frame].astype('float64') #filtreli gt verilerinden içinde bulunuğunuz kısım çıkarılıyor
                    gt_curr_frame = gt_curr_frame[:,2:6] #tlwh tipinde veriler alınıyor
                    #print(gt_curr_frame)
                    #print(self.my_tlwh_to_xywh(im, gt_curr_frame))
                    
                    
                    bbox_xywh = self.my_tlwh_to_xywh(im, gt_curr_frame) #yolo yerine gt bboxları 
                    cls_conf = np.ones((bbox_xywh.shape[0],), dtype=int) #yolo conf skorları yerine (tüm skorlar 1)
                    cls_ids = np.zeros(bbox_xywh.shape[0]) #bütün bboxlar yolo için 0 id'li yani person.
                    ori_im = draw_boxes(ori_im, self.my_tlwh_to_xyxy(im,gt_curr_frame)) #gt'deki bboxları çizdir
                    

                    print("yolo yerine gt kullanıldı, frame: ",idx_frame)
                    
                    #test amaçlı bilerek yanlış vererek başlangıçtaki verilerin tracker üzerindeki etkisini incelemek için
                    """
                    bbox_xywh = np.array([[100,200,400.1,600.1],[500,600.1,600.1,800.1]]) #test amaçlı bilerek yanlış vermek için
                    cls_conf = np.ones((bbox_xywh.shape[0],), dtype=int) #test amaçlı bilerek yanlış vermek için
                    cls_ids = np.zeros(bbox_xywh.shape[0]) #test amaçlı bilerek yanlış vermek için
                    ori_im = draw_boxes(ori_im, bbox_xywh)
                    """
                    
                    
                     
            
            """
            labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            """

            # select person class 0-people 22-zebra 20-elephant
            #mask = (cls_ids == 20) + (cls_ids == 22)
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im) #im.shape =  video_boyu,video_eni,3
            #print(bbox_xywh) # number_of_detection, 4
            #print(cls_conf) # number_of_detection,

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                
                #detection'ları ekrana çizen kendi yazdığım kod
                #ori_im = draw_boxes(ori_im, self.my_xywh_to_xyxy(im,bbox_xywh))
                
                #doğru eşleşmeleri ekrana çizen orjinal kod
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))
            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--gt", action="store_true") #gt'den alınan verileri kullanmak istiyorsak
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()
    


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
