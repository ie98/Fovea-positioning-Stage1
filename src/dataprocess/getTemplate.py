import os.path

import cv2
import pandas as pd
import numpy as np
import random
path = '../../dataset/Train/fundus_image/N0142.jpg'

img = cv2.imread(path)

print(img)
rootPath = '../../dataset/MESSIDOR/images'
def getImageAndLabel(imageNaem,label):
    img = cv2.imread(os.path.join(rootPath,imageNaem))
    label_x = label[0]
    label_y = label[1]
    new_label_x = (768/img.shape[1])*label_x
    new_label_y = (512/img.shape[0])*label_y
    img = cv2.resize(img,dsize=(768,512),interpolation=cv2.INTER_LINEAR)
    return img , new_label_x,new_label_y


excel = pd.read_excel('../../dataset/MESSIDOR/label/data.xls')
image_name = np.array(excel)[:,0]
labels = np.array(excel)[:,[2,3]]

# 使用 subset1 构建的模板  一共 568张 间隔 20 取一张 共 28 张
# sub_set_1 = [i for i in range(1136) if i % 2 == 0]
# sub_set_1 = np.array(sub_set_1)
# template_index = sub_set_1[ [i for i in range(568) if i % 20 == 0] ]

sub_set_2 = [i for i in range(1136) if i % 2 == 1]
sub_set_2 = np.array(sub_set_2)
template_index = sub_set_2[ [i for i in range(568) if i % 20 == 0] ]

name = image_name[template_index]

mask = labels[template_index]

template = np.zeros(shape=(64,64,3))

for i in range(len(name)):
    img,index_x,index_y = getImageAndLabel(name[i],mask[i,:])
    template = template + img[int(index_y-32):int(index_y+32),int(index_x-32):int(index_x+32),:]

template = template/len(name)
cv2.imwrite('../template_sub2_strid_20.jpg', template.astype(np.uint8))

# random_list = random.sample(range(0, len(image_name)), len(image_name))
# image_name = image_name[random_list]
# labels = labels[random_list]
#
#
#
# for i in range(0,30):
#     img,index_x,index_y = getImageAndLabel(image_name[i],labels[i,:])
#     template = template + img[int(index_y-64):int(index_y),int(index_x-64):int(index_x),:]
# template = template/30
# cv2.imwrite('../template_error.jpg', template.astype(np.uint8))


#
# # 截取 模板
# box_W = 200
# box_H = 200
#
# template = img[int(index_0_y-box_H/2):int(index_0_y+(box_H-box_H/2)),
#            int(index_0_x-box_W/2):int(index_0_x+(box_W-box_W/2)),:]
#
# temp_ing = cv2.imwrite('./temolate_{}_142.jpg'.format(box_W),template)
#
#
# img[int(index_0_y-box_H/2):int(index_0_y+(box_H-box_H/2)),
#            int(index_0_x-box_W/2):int(index_0_x+(box_W-box_W/2)),1] = 0
#
# cv2.imwrite('./mask_{}_142.jpg'.format(box_W),img)













