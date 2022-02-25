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
    img = img.half() if opt.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    if not opt.multi_scale:
        start = time.time()

    pred = model(img)[0]

    if not opt.multi_scale:
        forward_time = time.time() - start

    length = pred.shape[1]
    size_min = int(length/85)

    if not opt.multi_scale:
        start = time.time()

    pred1=[]
    pred1.append(pred[:,0:size_min*64])
    pred1.append(pred[:,size_min*64:size_min*80])
    pred1.append(pred[:,size_min*80:size_min*84])
    pred1.append(pred[:,size_min*84:size_min*85])

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

    if not opt.multi_scale:
        post_time = time.time() - start

    boxes = np.concatenate(boxes)

    if len(boxes) == 0:
        boxes = np.array([[0,0,0,0,0.001]])
    
    if opt.multi_scale:
        return boxes
    else:
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
        merge_index = np.where(o >= 0.6)[0]

        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        #if merge_index.shape[0] <= 1:
        #    continue
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

def multi_scale_test(opt, path, stride, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink

    img, img0 = load_image(path, stride, True, st)
    det_s = detect(model, img, img0, opt, True)

    if max_im_shrink > 0.75:
        img, img0 = load_image(path, stride, True, 0.75)
        det_s = np.row_stack((det_s, detect(model, img, img0, opt, True)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0], det_s[:, 3] - det_s[:, 1]) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    img, img0 = load_image(path, stride, True, bt)
    det_b = detect(model, img, img0, opt, True)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        img, img0 = load_image(path, stride, True, 1.5)
        det_b = np.row_stack((det_b, detect(model, img, img0, opt, True)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            img, img0 = load_image(path, stride, True, bt)
            det_b = np.row_stack((det_b, detect(model, img, img0, opt, True)))
            bt *= 2

        img, img0 = load_image(path, stride, True, max_im_shrink)
        det_b = np.row_stack((det_b, detect(model, img, img0, opt, True)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1]) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1]) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(opt, path, stride, max_shrink):
    img, img0 = load_image(path, stride, True, 0.25)
    det_b = detect(model, img, img0, opt, True)
    index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1])> 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            img, img0 = load_image(path, stride, True, st[i])
            det_temp = detect(model, img, img0, opt, True)
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0],
                               det_temp[:, 3] - det_temp[:, 1]) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0],
                               det_temp[:, 3] - det_temp[:, 1]) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/best1.33-1-val.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/WIDER_FACE/WIDER_val/images/', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path', default='D:/WIDER_FACE/val_results/val1.33-1/', help='save path')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--multi-scale', default=False, help='multi_scale_test')

    opt = parser.parse_args()
    print(opt)

    source, weights, save_path, multi_scale = opt.source, opt.weights, opt.save_path, opt.multi_scale

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if opt.half:
        model.half()  # to FP16

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
            max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5 # the max size of input image for caffe
            max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

            with torch.no_grad():
                if multi_scale:
                    img, img0 = load_image(path, stride, True)
                    preds1= detect(model, img, img0, opt, True)


                    preds2, preds3 = multi_scale_test(opt, path, stride, max_im_shrink)
                    preds4 = multi_scale_test_pyramid(opt, path, stride, max_im_shrink)
                
                    preds = np.r_[preds1, preds2, preds3, preds4]
                    preds = bbox_vote(preds)
                else:
                    img, img0 = load_image(path, stride)
                    preds = detect(model, img, img0, opt)
                    ft_sum = ft_sum+preds[1]
                    pt_sum = pt_sum+preds[2]
                    preds = preds[0]
                    start = time.time()
                    #preds = inverted_nms(preds, opt.iou_thres)
                    preds = bbox_vote(preds)
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

    if not multi_scale:
        print('Total Forward Time (s):', ft_sum)
        print('Total Post-Proc Time (s):', pt_sum)
            
