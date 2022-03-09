import argparse

import os
import numpy as np
import glob
import cv2
import time
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, xywh2xyxy
from utils.torch_utils import select_device
from inverted_nms import inverted_nms

def detect(model, img, im0s, opt, flip=False):
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    start = time.time()

    pred = model(img)[0]

    forward_time = time.time() - start

    length = pred.shape[1]
    size_min = int(length/21)

    start = time.time()

    pred1=[]
    pred1.append(pred[:,0:size_min*16])
    pred1.append(pred[:,size_min*16:size_min*20])
    pred1.append(pred[:,size_min*20:size_min*21])

    boxes=[]
    for j, pred in enumerate(pred1):
        if flip:
            pred[1,:,0] = img.shape[3] - pred[1,:,0]
            pred = torch.cat([pred[0], pred[1]], 0)
            pred = pred.unsqueeze(0)
        # Apply NMS
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)[0].cpu().numpy()

        pred = pred[0,:,:5].cpu().numpy()
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        pred = pred[pred[:,4]>opt.conf_thres]
        pred = inverted_nms(pred, opt.iou_thres)

        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape)

        boxes.append(pred[:, :5])

    post_time = time.time() - start

    boxes = inverted_nms(np.concatenate(boxes), opt.iou_thres)

    if len(boxes)==0:
        boxes = np.array([[0,0,0,0,0.001]])
    
    return boxes, forward_time, post_time

def load_image(path, stride, flip=False, shrink=1):
    # Read image
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path

    img_size = max(img0.shape[:2])
    img_size = int(np.round(img_size*shrink))
    img_size = check_img_size(img_size, s=stride)

    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1]#.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    if flip:
        img = np.array([img, cv2.flip(img,1)])
        img = img.transpose(0, 3, 1, 2) # to 3x416x416
    else:
        img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    return img, img0

def write_txt(path, preds):
    f = open(path, 'w')
    path = path.split('/')
    f.write(path[-1][:-4]+'\n')
    n = len(preds)
    f.write(str(n)+'\n')
    for i in range(n):
        box = preds[i]
        f.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(box[4])+'\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/face_yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/WIDER_FACE/WIDER_val/images/', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path', default='val_results/face_yolov5x/', help='save path')

    opt = parser.parse_args()
    print(opt)

    source, weights, save_path = opt.source, opt.weights, opt.save_path

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 32, 32).to(device).type_as(next(model.parameters())))  # run once

    events = os.listdir(opt.source)

    ft_sum = 0
    pt_sum = 0
    im_sum = 0

    for event in events:
        paths = sorted(glob.glob(opt.source+event+'/*', recursive=True))

        for img_num, path in enumerate(paths):
            print(event, img_num, path)
            im_sum = im_sum+1

            img = cv2.imread(path)

            with torch.no_grad():
                img, img0 = load_image(path, stride)
                preds = detect(model, img, img0, opt)
                ft_sum = ft_sum+preds[1]
                pt_sum = pt_sum+preds[2]
                preds = preds[0]
                start = time.time()
                pt_sum = pt_sum+time.time()-start
                
                preds[:,2] = preds[:,2]-preds[:,0]
                preds[:,3] = preds[:,3]-preds[:,1]

                filename = os.path.basename(path)
                path_save = save_path+event
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
                path_txt = path_save+'/'+filename[:-3]+'txt'
                write_txt(path_txt, preds)

    print('Total Images:', im_sum)

    print('Total Forward Time (s):', ft_sum)
    print('Total Post-Proc Time (s):', pt_sum)
            
