import os
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from yolov3_deepsort import VideoTracker
from utils.parser import get_config

import motmetrics as mm
mm.lap.default_solver = 'lap'
from utils.evaluation import Evaluator

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def main(data_root='', seqs=('',), args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    


    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    # run tracking
    accs = []
    for seq in seqs:
        result_root = os.path.join(Path(data_root), "{}".format(seq)) #result klasörü her seq için değişiyor
        logger.info('start seq: {}'.format(seq)) 
        result_filename = os.path.join(result_root, 'results2.txt') #MOT16/train_ya_da_test/MOT16-XX/results.txt (input txt, yolov3_deepsort.py üretecek, yolov3_deepsort_eval.py üretilen dosyayı değerlendiriken gt ile birlikte okuyacak)
        video_path = data_root+"/"+seq+"/v.mp4" # MOT16/train_ya_da_test/MOT16-XX/v.mp4 (input mp4, yolov3_deepsort.py okuyacak)
        #print(result_filename)
        
        #argumanlar burda tekrar alınıyor. çünkü save_path=result_root her video seq için değişik olmalı
        args = parse_args(save_path=result_root)

        #with VideoTracker(cfg, args, video_path) as vdo_trk:
        #    vdo_trk.run()

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type) #gt şu adresten okunuyor data_root/seq/gt/gt.txt
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(data_root, 'summary_global.xlsx'))


def parse_args(save_path=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--gt", action="store_true") #gt'den alınan verileri kullanmak istiyorsak
    parser.add_argument("--display_height", type=int, default=600)
    #her result videosu ve her result.txt seqs_str ile belirtilen klasörün içine yazılıyor. (yolov3_deepsort.py tarafından)
    #Bu parametre yolov3_deepsort.py çağırılırken her video sequence de otomatik olarak değiştiriliyor.
    #bundan dolayı bu parametreyi terminalden almaya gerek yok.
    parser.add_argument("--save_path", type=str, default=save_path) 
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':
    #INFO:
    #Bu .py dosyası videoları MOT16 kriterlerine göre değerlendiriyor.
    #Önce yolov3_deepsort.py çağırılıp ilgili video için outputlar üretiliyor. (results.avi ve results.txt)
    #daha sonra results.txt ve gt.txt kıyaslanarak mot skorları hesaplanıyor ve summary_global.xlsx ismiyle kaydediliyor.



    args = parse_args() #save_path argumanı her video frame'inde değişeceği için daha sonra tekrar alınacak

    #orj kod:
    
    seqs_str = '''MOT16-02       
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13
                  '''        
    
    
    """    
    seqs_str = '''MOT16-04
                  MOT16-05
                  MOT16-10
                  '''
    """
    
    #olması gereken klasör düzeni:
    #MOT16/train/MOT16-XX/
    #                    /gt/gt.txt
    #                    /v.mp4
    #                    /results.txt (üretilecek)
    #                    /results.avi (üretilecek)
    #           /summary_global.xlsx (üretilecek)
    data_root = 'MOT16/train'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs,
         args=args)