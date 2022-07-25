import os.path
import time
import cv2
import pandas as pd
import numpy as np
import random

def timelode(scale):
    print("执行开始，祈祷不报错".center(scale // 2, "-"))
    start = time.perf_counter()
    for i in range(scale + 1):
        a = "*" * i
        b = "." * (scale - i)
        c = (i / scale) * 100
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
        time.sleep(0.1)

    print("\n" + "执行结束，万幸".center(scale // 2, "-"))

def loadImgs():
    path = '../../dataset/Train/fundus_image/H0001.jpg'
    root_path = '../../dataset/Train/fundus_image'
    img = cv2.imread(path)

    print(img)

    excel = pd.read_excel('../../dataset/Train/Fovea_Location_train.xlsx')

    print(excel)
    excel = np.array(excel)
    label = excel[:,[1,2]]

    image_names = excel[:,0]

    imgs = np.ndarray(shape=[image_names.shape[0],1444,1444,3])
    dict = {'1444*1444':0}
    for i in range(image_names.shape[0]):
        img = cv2.imread(os.path.join(root_path,str(image_names[i])))
        if img.shape[0] != 1444 or img.shape[1] != 1444:
            print('{} size error !!'.format(image_names[i]))
            key = str(img.shape[0])+'*'+str(img.shape[1])
            if dict.__contains__(key):
                dict[key] = dict[key] + 1

            else:
                dict[key] = 1
        else:
            imgs[i,:] = img
            dict['1444*1444'] = dict['1444*1444'] + 1
        print(i)
        if i == 799:
            print('all are right !')
    print(dict)

def randomAndSave(space,name):
    randomlist = random.sample(range(0, space.shape[0]), space.shape[0])
    space = space[randomlist]
    dataframe = pd.DataFrame(
        {'image_name': space[:,0],
         'pointRate_col': space[:,1],
         'label_x':space[:,2],
         'label_y':space[:,3]
         })
    dataframe.to_csv('./{}.csv'.format(name), sep=',')
    return space
def TVTsplit(train_list,val_list,test_list,space):
    val_list = np.concatenate((val_list,space[:int(space.shape[0]/10),:]),axis=0)
    test_list = np.concatenate((test_list,space[int(space.shape[0]/10):int(space.shape[0]/5),:]),axis=0)
    train_list = np.concatenate((train_list,space[int(space.shape[0]/5):,:]),axis=0)
    return train_list,val_list,test_list

def saveFile(file,name):
    dataframe = pd.DataFrame(
        {'image_name': file[:, 0],
         'pointRate_col': file[:, 1],
         'label_x': file[:, 2],
         'label_y': file[:, 3]
         })
    dataframe.to_csv('./{}.csv'.format(name), sep=',')

def loadLabel():
    path = '../../dataset/Train/fundus_image/H0001.jpg'
    root_path = '../../dataset/Train/fundus_image'
    img = cv2.imread(path)

    print(img)

    excel = pd.read_excel('../../dataset/Train/Fovea_Location_train.xlsx')

    print(excel)
    excel = np.array(excel)
    label = excel[:, [1, 2]]  # 标注是 [列,行] cv2读取是 [行,列,3]

    image_names = excel[:, 0]

    pointRate_col = []
    pointRate_row = []
    for i in range(image_names.shape[0]):
    # for i in range(30):
        image = cv2.imread(os.path.join(root_path,str(image_names[i])))
        curr_pointRate_col = int(label[i,0])/image.shape[1]
        pointRate_col.append(curr_pointRate_col)
        curr_pointRate_row = int(label[i,1])/image.shape[0]
        pointRate_row.append(curr_pointRate_row)


    pointRate_col = np.array(pointRate_col)
    pointRate_row = np.array(pointRate_row)
    imageDetail = image_names
    imageDetail = np.concatenate((imageDetail.reshape(-1,1),pointRate_col.reshape(-1,1),label),axis=1)
    # 数据划分 根据 pointRate_col 划分为 6 个区间
    space_1 = None  # 1-0.9
    space_2 = None  # 0.9-0.8
    space_3 = None  # 0.8-0.7
    space_4 = None  # 0.7-0.6
    space_5 = None  # 0.6-0.5
    space_6 = None  # 0.5-0.4

    for i in range(imageDetail.shape[0]):
        if float(imageDetail[i,1]) > 0.9 and float(imageDetail[i,1]) <= 0.9375:
            if space_1 is None:
                space_1 = imageDetail[i].reshape(1,4)
            else:
                space_1 = np.concatenate((space_1,imageDetail[i].reshape(1,4)),0)

        if float(imageDetail[i,1]) > 0.8 and float(imageDetail[i,1]) <= 0.9:
            if space_2 is None:
                space_2 = imageDetail[i].reshape(1,4)
            else:
                space_2 = np.concatenate((space_2,imageDetail[i].reshape(1,4)),0)

        if float(imageDetail[i,1]) > 0.7 and float(imageDetail[i,1]) <= 0.8:
            if space_3 is None:
                space_3 = imageDetail[i].reshape(1,4)
            else:
                space_3 = np.concatenate((space_3,imageDetail[i].reshape(1,4)),0)

        if float(imageDetail[i,1]) > 0.6 and float(imageDetail[i,1]) <= 0.7:
            if space_4 is None:
                space_4 = imageDetail[i].reshape(1,4)
            else:
                space_4 = np.concatenate((space_4,imageDetail[i].reshape(1,4)),0)

        if float(imageDetail[i,1]) > 0.5 and float(imageDetail[i,1]) <= 0.6:
            if space_5 is None:
                space_5 = imageDetail[i].reshape(1,4)
            else:
                space_5 = np.concatenate((space_5,imageDetail[i].reshape(1,4)),0)

        if float(imageDetail[i,1]) > 0.3 and float(imageDetail[i,1]) <= 0.5:
            if space_6 is None:
                space_6 = imageDetail[i].reshape(1,4)
            else:
                space_6 = np.concatenate((space_6,imageDetail[i].reshape(1,4)),0)




    # dataframe = pd.DataFrame(
    #     {'image_name': image_names.reshape(-1),
    #      'pointRate_col': pointRate_col,
    #      'pointRate_row': pointRate_row
    #      })
    # dataframe.to_csv('./pointRate.csv', sep=',')

    space_1 = randomAndSave(space_1, 'space_1')
    space_2 = randomAndSave(space_2, 'space_2')
    space_3 = randomAndSave(space_3, 'space_3')
    space_4 = randomAndSave(space_4, 'space_4')
    space_5 = randomAndSave(space_5, 'space_5')
    space_6 = randomAndSave(space_6, 'space_6')

    # 初始化
    val_list = space_1[:int(space_1.shape[0]/10),:]
    test_list = space_1[int(space_1.shape[0]/10):int(space_1.shape[0]/5),:]
    train_list = space_1[int(space_1.shape[0]/5):,:]

    train_list,val_list,test_list = TVTsplit(train_list=train_list,val_list=val_list,test_list=test_list,space=space_2)

    train_list, val_list, test_list = TVTsplit(train_list=train_list, val_list=val_list, test_list=test_list,space=space_3)

    train_list, val_list, test_list = TVTsplit(train_list=train_list, val_list=val_list, test_list=test_list,space=space_4)

    train_list, val_list, test_list = TVTsplit(train_list=train_list, val_list=val_list, test_list=test_list,space=space_5)

    train_list, val_list, test_list = TVTsplit(train_list=train_list, val_list=val_list, test_list=test_list,space=space_6)

    saveFile(train_list,'train_list')
    saveFile(val_list,'val_list')
    saveFile(test_list,'test_list')











if __name__ == '__main__':
    loadLabel()
















