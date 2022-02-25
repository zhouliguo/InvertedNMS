import argparse
import time

import numpy as np
import cv2
import torch
import torchvision

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from utils.torch_utils import select_device
from utils.metrics import box_iou

def inverted_nms(det):
    zero_index = np.where((det[:,2] <= det[:,0]) | (det[:,3] <= det[:,1]))[0]
    det = np.delete(det, zero_index, 0)

    order = det[:, 4].ravel().argsort()
    det = det[order, :]
    dets = np.zeros((0, 5), dtype=np.float32)

    area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])

    for i in range(len(det)-1):
        boxes = det[0:i+1]
        
        xx1 = np.maximum(det[i+1, 0], boxes[:, 0])
        yy1 = np.maximum(det[i+1, 1], boxes[:, 1])
        xx2 = np.minimum(det[i+1, 2], boxes[:, 2])
        yy2 = np.minimum(det[i+1, 3], boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        o = inter / (area[i+1] + area[0:i+1] - inter)
        # get needed merge det and delete these det
        iou = np.where(o > 0.6)[0]
        det[iou,:] = 0

    zero_index = np.where((det[:,4] == 0))
    det = np.delete(det, zero_index, 0)

    return det

def detect(model, img, im0s, opt, flip=False):
    img = torch.from_numpy(img).to(device)
    img = img.half() if opt.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    if flip:
        pred[:,:,0] = img.shape[3] - pred[:,:,0] - 1

    length = pred.shape[1]
    size_min = int(length/85)

    pred1=[]
    pred1.append(pred[:,0:size_min*64])
    pred1.append(pred[:,size_min*64:size_min*80])
    pred1.append(pred[:,size_min*80:size_min*84])
    pred1.append(pred[:,size_min*84:size_min*85])

    boxes=[]
    for j, pred in enumerate(pred1):
        # Apply NMS
        pred_nms = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)[0].cpu().numpy()
        #pred = pred[0,:,:5].cpu().numpy()
        #pred[:, :4] = xywh2xyxy(pred[:, :4])
        #pred = pred[pred[:,4]>opt.conf_thres]
        #pred = inverted_nms(pred)

        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape)

        boxes.append(pred[:, :5])

    if len(boxes)==0:
        return np.array([[0,0,0,0,0.001]])
    return np.concatenate(boxes)

def load_image(path, stride, flip=False, shrink=1):
    # Read image
    img0 = cv2.imread(path)  # BGR
    img_size = max(img0.shape[:2])
    img_size = img_size*shrink
    img_size = check_img_size(img_size, s=stride)
    if flip:
        img0 = cv2.flip(img0,1)
    assert img0 is not None, 'Image Not Found ' + path

    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, img0

def bbox_vote(det):
    zero_index = np.where((det[:,2] <= det[:,0]) | (det[:,3] <= det[:,1]))[0]
    det = np.delete(det, zero_index, 0)

    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = np.zeros((0, 5),dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o > 0.6)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    return dets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/weight_light.pt', help='model.pt path(s)')
    parser.add_argument('--image-path', type=str, default='figures/0_Parade_marchingband_1_364.jpg', help='image')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)

    weights, image_path = opt.weights, opt.image_path

    # Initialize
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if opt.half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 32, 32).to(device).type_as(next(model.parameters())))  # run once

    with torch.no_grad():
        img, img0 = load_image(image_path, stride, False)
        preds = detect(model, img, img0, opt, False)
        preds = bbox_vote(preds).astype(np.float32)

        for pred in preds:
            if pred[4]>=0.5:
                pred = np.round(pred).astype(np.int32)
                cv2.rectangle(img0, (pred[0], pred[1]), (pred[2], pred[3]), (0,255,0), 2)
        #cv2.imwrite('widerimage.png', img0)
        cv2.imshow('image', img0)
        cv2.waitKey()

 
