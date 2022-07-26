import math
import os
import numpy as np
import pandas as pd

class computerRrate_test:
    def __init__(self,num):
        self.subSet_Details = np.array(pd.read_csv('./getDetail/subSet_{}_Detail.csv'.format(num)))

    def computer(self,preRes,saveFilePath,epoch,valImgNames):

        # preRes = np.array(pd.read_csv('./val_detail_248.csv'))
        resDict = {}
        R_rate_list = []
        up_x_list = []
        up_y_list = []
        real_dist_list = []
        count = 0
        for i in range(len(preRes)):
            if valImgNames[i] != self.subSet_Details[i, 1]:
                print('image name error !!!')
            label_x = self.subSet_Details[i, 5]
            label_y = self.subSet_Details[i, 6]
            img_x = self.subSet_Details[i, 2]
            img_y = self.subSet_Details[i, 3]
            R = self.subSet_Details[i, 4]
            pre_x = preRes[i, 0]
            pre_y = preRes[i, 1]
            up_x = pre_x * (img_x / 512)
            up_y = pre_y * (img_y / 512)

            real_dist = math.sqrt((up_x - label_x) ** 2 + (up_y - label_y) ** 2)

            real_dist_list.append(real_dist)

            r_rate = real_dist / R
            if r_rate <= 0.125:
                count = count + 1
            R_rate_list.append(r_rate)

        allDetails = {
            'name': self.subSet_Details[:, 1],
            'W': self.subSet_Details[:, 2],
            'H': self.subSet_Details[:, 3],
            'dist': real_dist_list,
            'R': self.subSet_Details[:, 4],
            'R_rate': R_rate_list
        }
        if os.path.exists(saveFilePath) is False:
            os.makedirs(saveFilePath)
        dataframe = pd.DataFrame(allDetails)

        dataframe.to_csv('{}/{}_R_rate.csv'.format(saveFilePath,epoch), sep=',')

        return count






class computerRrate_val:
    def __init__(self,num):
        val_indexs = [i for i in range(568) if i % 5 == 0]
        self.subSet_1_Details = np.array(pd.read_csv('./getDetail/subSet_{}_Detail.csv'.format(num)))[val_indexs,:]

    def computer(self,preRes,saveFilePath,epoch,valImgNames):

        # preRes = np.array(pd.read_csv('./val_detail_248.csv'))
        resDict = {}
        R_rate_list = []
        up_x_list = []
        up_y_list = []
        real_dist_list = []
        count = 0
        for i in range(len(preRes)):
            if valImgNames[i] != self.subSet_1_Details[i, 1]:
                print('image name error !!!')
            label_x = self.subSet_1_Details[i, 5]
            label_y = self.subSet_1_Details[i, 6]
            img_x = self.subSet_1_Details[i, 2]
            img_y = self.subSet_1_Details[i, 3]
            R = self.subSet_1_Details[i, 4]
            pre_x = preRes[i, 0]
            pre_y = preRes[i, 1]
            up_x = pre_x * (img_x / 512)
            up_y = pre_y * (img_y / 512)

            real_dist = math.sqrt((up_x - label_x) ** 2 + (up_y - label_y) ** 2)

            real_dist_list.append(real_dist)

            r_rate = real_dist / R
            if r_rate <= 0.125:
                count = count + 1
            R_rate_list.append(r_rate)

        allDetails = {
            'name': self.subSet_1_Details[:, 1],
            'W': self.subSet_1_Details[:, 2],
            'H': self.subSet_1_Details[:, 3],
            'dist': real_dist_list,
            'R': self.subSet_1_Details[:, 4],
            'R_rate': R_rate_list
        }
        if os.path.exists(saveFilePath) is False:
            os.makedirs(saveFilePath)
        dataframe = pd.DataFrame(allDetails)

        dataframe.to_csv('{}/{}_R_rate.csv'.format(saveFilePath,epoch), sep=',')

        return count





#
# if __name__ == '__main__':
#     subSet_2_Details = np.array(pd.read_csv('./subSet_2_Detail.csv'))
#     preRes = np.array(pd.read_csv('./val_detail_248.csv'))
#     resDict = {}
#     R_rate_list = []
#     up_x_list = []
#     up_y_list = []
#     real_dist_list = []
#     for i in range(len(preRes)):
#         if preRes[i,1] != subSet_2_Details[i,1]:
#             print('image name error !!!')
#         label_x = subSet_2_Details[i,5]
#         label_y = subSet_2_Details[i,6]
#         img_x = subSet_2_Details[i,2]
#         img_y = subSet_2_Details[i,3]
#         R = subSet_2_Details[i,4]
#         pre_x = preRes[i,4]
#         pre_y = preRes[i,5]
#         up_x = pre_x * (img_x/512)
#         up_y = pre_y * (img_y/512)
#
#         real_dist = math.sqrt((up_x-label_x)**2+(up_y-label_y)**2)
#
#         real_dist_list.append(real_dist)
#
#         r_rate = real_dist/R
#         R_rate_list.append(r_rate)
#
#     allDetails = {
#         'name':subSet_2_Details[:,1],
#         'W':subSet_2_Details[:,2],
#         'H':subSet_2_Details[:,3],
#         'dist':real_dist_list,
#         'R':subSet_2_Details[:,4],
#         'R_rate':R_rate_list
#     }
#     dataframe = pd.DataFrame(allDetails)
#
#     dataframe.to_csv('./subSet2R_rate.csv', sep=',')
#
#
