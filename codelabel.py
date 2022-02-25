import os
import cv2
import numpy as np

def codelabel(path, image_list):
    images = []
    txts = []
    txts2 = []
    folderlist = os.listdir(path+'labels')

    fl = open(image_list,'w')
    for folder in folderlist:
        filelist = os.listdir(os.path.join(path+'labels', folder))
        for l in filelist:
            fl.write(path+'images/'+folder+'/'+l[:-3]+'jpg\n')
            images.append(path+'images/'+folder+'/'+l[:-3]+'jpg')
            txts.append(path+'labels/'+folder+'/'+l)
            txts2.append(path+'label/'+folder+'/'+l)
    fl.close()

    for txt, txt2, image in zip(txts, txts2, images):
        if not os.path.exists(os.path.dirname(txt2)):
            os.makedirs(os.path.dirname(txt2))
        img = cv2.imread(image)
        h, w = img.shape[:2]
        f = open(txt, 'r')
        f2 = open(txt2, 'w')

        lines = f.readlines()
        for l in lines:
            bbox = l.strip('\n').split(' ')
            bbox = np.array(bbox).astype(np.float64)

            bbox[0] = (bbox[0]+bbox[2]/2)/w
            bbox[1] = (bbox[1]+bbox[3]/2)/h
            bbox[2] = bbox[2]/w
            bbox[3] = bbox[3]/h
            bbox[bbox>1] = 1
            f2.write('0 '+str(round(bbox[0],6))+' '+str(round(bbox[1],6))+' '+str(round(bbox[2],6))+' '+str(round(bbox[3],6))+'\n')
        f.close()
        f2.close()

if __name__ == '__main__':
    # path is path of subset
    # path + 'labels' is the original label folder
    # path + 'label' is the coded label folder
    # image_list is the image path list
    codelabel(path = 'D:/WIDER_FACE/WIDER_test/', image_list='test.txt')