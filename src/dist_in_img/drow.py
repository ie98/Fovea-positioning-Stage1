import PIL.Image as Image
import cv2
import numpy as np
import pandas as pd
import os

def getMaskboxImg(imgDetil:dict):
    imgName = imgDetil['imgName']
    imgpath = imgDetil['imgpath']
    label_col = imgDetil['label_col']
    label_row = imgDetil['label_row']
    pre_col = imgDetil['pre_col']
    pre_row = imgDetil['pre_row']
    preImg_col = imgDetil['preImg_col']
    preImg_row = imgDetil['preImg_row']
    dist = imgDetil['dist']


    img = cv2.imread(imgpath)
    cv2.rectangle(img, (int(label_col-10),int(label_row-10)),(int(label_col+10),int(label_row+10)),(0,255,0),2)

    # cv2 读取图像文件后的格式是 【行row ，列col，3通道】
    # 放大到原图的标记列
    pre_up_col = (pre_col/preImg_col) * img.shape[1]
    # 放大到原图的标记行
    pre_up_row = (pre_row/preImg_row) * img.shape[0]
    cv2.rectangle(img, (int(pre_up_col-10),int(pre_up_row-10)),(int(pre_up_col+10),int(pre_up_row+10)),(0,0,255),2)

    cv2.line(img,(int(label_col),int(label_row)),(int(pre_up_col),int(pre_up_row)),(255,0,0),2)
    cv2.putText(img, imgName+' :'+str(dist), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)



    cv2.imwrite('./preImg/{}.png'.format(imgName),img)

def getLabels(labelFilesName):
    basePath = '../../dataset/MESSIDOR'
    labels = pd.read_excel(os.path.join(basePath, 'label', labelFilesName))
    return labels

if __name__ == '__main__':

    predDist = pd.read_csv('preDist/val_detail_136.csv')
    predDist_np = np.array(predDist)
    # label = pd.read_excel('../../dataset/Train/Fovea_Location_train.xlsx')
    # label_np = np.array(label)
    labelFileName = 'data.xls'
    labelDetail = np.array(getLabels(labelFileName))
    imgNames = labelDetail[:,0]
    labels = labelDetail[:, [0,2, 3]]
    sub_set_2 = [i for i in range(1136) if i % 2 == 1]
    val_labels = labels[sub_set_2]

    for i in range(predDist_np.shape[0]):
        imgDetil = {}
        imgDetil['imgName'] = predDist_np[i,1]
        imgDetil['imgpath'] = '../../dataset/MESSIDOR/images/{}'.format(predDist_np[i,1])
        index = np.where(labels[:,0] == predDist_np[i, 1])
        imgDetil['label_col'] = float(val_labels[index,1])
        imgDetil['label_row'] = float(val_labels[index,2])
        imgDetil['pre_col'] = float(predDist_np[i,4])
        imgDetil['pre_row'] = float(predDist_np[i,5])
        imgDetil['preImg_col'] = 512
        imgDetil['preImg_row'] = 512
        imgDetil['dist'] = float(predDist_np[i,6])

        getMaskboxImg(imgDetil)
        print('{} is Done.'.format(i))






