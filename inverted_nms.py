import numpy as np

def inverted_nms(det, iou_thres):
    zero_index = np.where((det[:,2] <= det[:,0]) | (det[:,3] <= det[:,1]))[0]
    det[zero_index,:] = 0

    order = det[:, 4].ravel().argsort()
    det = det[order, :]

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

        iou = np.where(o > iou_thres)[0]
        if len(iou)==0:
            continue
        det[iou,:] = 0

    index = np.where((det[:,4] != 0))

    return det[index]