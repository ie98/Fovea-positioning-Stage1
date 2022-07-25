import numpy as np
import pandas as pd
import os
import cv2
basePath = '../../dataset/MESSIDOR'


def getImg(imgName):
    img = cv2.imread(os.path.join(basePath,'images', imgName), -1)
    return img


labelFileName = 'data.xls'
def getLabels(labelFilesName):
    labels = pd.read_excel(os.path.join(basePath, 'label', labelFileName))
    return labels


def getAllImgSize(imgNames,saveFileName,labels):

    imageDetail = {}

    R_dict = {
        '1440':68,
        '2240':103,
        '2304':109
    }
    name_list = []
    x_list = []
    y_list = []
    r_list = []
    label_x = []
    label_y = []
    for i in range(len(imgNames)):
        currImg = cv2.imread(os.path.join(basePath,'images',imgNames[i]))
        name_list.append(imgNames[i])
        x_list.append(currImg.shape[1])
        y_list.append(currImg.shape[0])
        r_list.append(R_dict[str(currImg.shape[1])])
        label_x.append(labels[i,0])
        label_y.append(labels[i,1])

    imageDetail['name'] = name_list
    imageDetail['real_x'] = x_list
    imageDetail['real_y'] = y_list
    imageDetail['R'] = r_list
    imageDetail['label_x'] = label_x
    imageDetail['label_y'] = label_y
    dataframe = pd.DataFrame(imageDetail)

    dataframe.to_csv('./{}.csv'.format(saveFileName), sep=',')

if __name__ == '__main__':
    labelFileName = 'data.xls'
    labelDetail = np.array(getLabels(labelFileName))
    imgNames = labelDetail[:, 0]
    labels = labelDetail[:,[2,3]]
    randomSampleFile = pd.read_csv('randomSample.csv')
    randomSampleId = np.array(randomSampleFile)[:, 1]
    imgNames = imgNames[randomSampleId]
    labels = labels[randomSampleId]
    sub_set_1 = [i for i in range(1136) if i % 2 == 0]
    sub_set_2 = [i for i in range(1136) if i % 2 == 1]
    subset1_names = imgNames[sub_set_1]
    subset1_labels = labels[sub_set_1]
    getAllImgSize(subset1_names,'subSet_1_Detail_randomSample',subset1_labels)
    subset2_names = imgNames[sub_set_2]
    subset2_labels = labels[sub_set_2]
    getAllImgSize(subset2_names,'subSet_2_Detail_randomSample',subset2_labels)