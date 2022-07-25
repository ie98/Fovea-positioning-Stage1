import cv2
import os
import pandas as pd
import numpy as np
basePath = '../../dataset/MESSIDOR'



labelFileName = 'data.xls'

def getLabels(labelFilesName):
    labels = pd.read_excel(os.path.join(basePath, 'label', labelFileName))
    return labels


valset = pd.read_csv('val_detail_1.csv')

valImgName =np.array(valset)[:,1]

allLabel = getLabels(labelFileName)

allLabel = np.array(allLabel)[:,[0,2,3]]
subSet_1 = None
subSet_2 = None

for i in range(len(allLabel)):
    flag = 0
    curr_img_name = allLabel[i,0]
    for j in range(len(valImgName)):

        if curr_img_name == valImgName[j]:
            if subSet_2 is None:
                subSet_2 = allLabel[i].reshape(1,-1)
            else:
                subSet_2 = np.concatenate((subSet_2,allLabel[i].reshape(1,-1)),axis=0)
            flag = 1
            break
    if flag == 0:
        if subSet_1 is None:
            subSet_1 = allLabel[i].reshape(1,-1)
        else:
            subSet_1 = np.concatenate((subSet_1,allLabel[i].reshape(1,-1)),axis=0)

print(123456)

dataframe = pd.DataFrame(
                    {'image_name': subSet_1[:,0], 'label_site_x': subSet_1[:,1],
                     'label_site_y': subSet_1[:, 2]})

dataframe.to_csv('./subSet_1.csv', sep=',')

dataframe = pd.DataFrame(
    {'image_name': subSet_2[:, 0], 'label_site_x': subSet_2[:, 1],
     'label_site_y': subSet_2[:, 2]})

dataframe.to_csv('./subSet_2.csv', sep=',')

