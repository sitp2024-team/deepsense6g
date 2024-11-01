# -*- coding: utf-8 -*-

###########################################
#  这段代码是对现有数据进行处理，生成不同csv
###########################################
import os
import numpy as np
import pandas as pd
import random
import itertools
import pickle
import ast
import matplotlib.pyplot as plt
from matplotlib import colors

from numpy.random import RandomState
from sklearn.utils import shuffle

import argparse

###############################################
#### Input dataset name
# 设定数据集名称和路径：
# 通过变量scenario_num定义了数据集的编号，接着使用这个编号构造出
# 数据集的根文件夹名称root_folder和数据文件的路径data_csv
###############################################
scenario_num = 5
# root_folder = f'Scenario{scenario_num}'
root_folder = f'/Data/Scenario{scenario_num}/development_dataset'
saved_img_folder = f'/Solution/Scenario{scenario_num}/image_beam'
saved_img_pos_folder = f'/Solution/Scenario{scenario_num}/img_pos_beam'
saved_pos_folder = f'/Solution/Scenario{scenario_num}/pos_beam'
# data_csv = f'./Data/Scenario{scenario_num}/development_dataset/scenario{scenario_num}_dev_train.csv'
# 修改数据集后
# data_csv = f'./Data/Scenario{scenario_num}/development_dataset/scenario{scenario_num}_dev.csv'
#加入假数据后
data_csv = f'./Data/Scenario{scenario_num}/development_dataset/scenario{scenario_num}_dev_include_fake.csv'

###############################################
# Read dataset to create a list of the input sequence
# 读取数据集：使用Pandas库读取CSV文件
# 从中提取了图像数据、功率数据和原始波束索引数据
###############################################

df = pd.read_csv(data_csv)
# 之前写的image_data_lst = df['unit2_loc_1'].values
image_data_lst = df['unit1_rgb_1'].values
pwr_data_lst = df['unit1_pwr_1'].values

# original_beam = df['unit1_beam_index'].values


###############################################
#### subsample the power and generate the
#### updated beam indices
# 处理功率数据和更新波束索引：
# 对于每个功率数据条目，读取相应的功率数据文件。
# 计算原始波束索引为功率数据中最大值的索引。
# 对功率数据进行二次采样（每隔一个取一个数据），基于更新后的功率数据重新计算波束索引。
###############################################
updated_beam = []
original_beam = []
for entry in pwr_data_lst:
    print(entry)
    ##两个路径共同组合成毫米波的路径
    data_to_read = f'./{root_folder}{entry[1:]}'
    pwr_data = np.loadtxt(data_to_read)
    ##计算最大值的索引
    original_beam.append(np.argmax(pwr_data) + 1)
    updated_pwr = []
    j = 0
    while j < (len(pwr_data) - 1):
        tmp_pwr = pwr_data[j]
        updated_pwr.append(tmp_pwr)
        j += 2
    updated_beam.append(np.argmax(updated_pwr) + 1)


# 创建图像-波束数据集：
# 根据图像数据列表更新图像路径。
# 创建一个新的DataFrame，包含更新后的图像路径、更新后的波束索引和原始波束索引。
# 将DataFrame分割为训练集、验证集和测试集，并保存为CSV文件。
def create_img_beam_dataset():
    # 确定文件保存位置
    ##folder_to_save = f'./ML_code/{root_folder}/image_beam'
    # folder_to_save = f'./{root_folder}/image_beam'
    # if not os.path.exists(folder_to_save):
    #     os.makedirs(folder_to_save)

    #############################################
    ###### created updated image path ###########
    #############################################
    updated_img_path = []
    for entry in image_data_lst:
        # 分割路径选择第二部分
        # 即./unit2/GPS_data/gps_location_711.txt取unit2/GPS_data/gps_location_711.txt
        img_path = entry.split('./')[1]
        updated_path = f'../../../{root_folder}/{img_path}'
        updated_img_path.append(updated_path)

    #############################################
    # 创建DataFrame以及其四个属性
    # 分别存储索引、更新后的图像路径，更新后的波束索引和原始波束索引
    #############################################

    indx = np.arange(1, len(updated_beam) + 1, 1)
    df_new = pd.DataFrame()
    df_new['index'] = indx
    df_new['unit1_rgb'] = updated_img_path
    df_new['unit1_beam_32'] = updated_beam
    df_new['unit1_beam_64'] = original_beam
    df_new.to_csv(fr'./{saved_img_folder}/scenario{scenario_num}_img_beam.csv', index=False)

    #############################################
    # 对以上数据集进行重排拆分
    # 获得新的train,val,test,比例为6：3：1
    #############################################
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng), [int(.6 * len(df_new)), int(.9 * len(df_new))])
    train.to_csv(f'./{saved_img_folder}/scenario{scenario_num}_img_beam_train.csv', index=False)
    val.to_csv(f'./{saved_img_folder}/scenario{scenario_num}_img_beam_val.csv', index=False)
    test.to_csv(f'./{saved_img_folder}/scenario{scenario_num}_img_beam_test.csv', index=False)


# 创建位置-波束数据集：
# 读取位置数据，对经纬度进行归一化处理。
# 创建一个新的DataFrame，包含归一化后的位置数据、更新后的波束索引和原始波束索引。
# 分割数据集为训练集、验证集和测试集，并保存为CSV文件。
def create_pos_beam_dataset():
    ##folder_to_save = f'./ML_code/{root_folder}/pos_beam'
    # folder_to_save = f'./{root_folder}/pos_beam'
    # if not os.path.exists(folder_to_save):
    #     os.makedirs(folder_to_save)

    ###############################################
    ####### read position values from dataset #####
    ###############################################

    lat = []
    lon = []

    ###pos_data_path = df['unit2_loc'].values
    pos_data_path = df['unit2_loc_1'].values
    for entry in pos_data_path:
        data_to_read = f'./{root_folder}{entry[1:]}'
        pos_val = np.loadtxt(data_to_read)
        # lat_val, lon_val = pos_val[0], pos_val[1]
        lat.append(pos_val[0])
        lon.append(pos_val[1])

    def norm_data(data_lst):
        norm_data = []
        for entry in data_lst:
            norm_data.append((entry - min(data_lst)) / (max(data_lst) - min(data_lst)))
        return norm_data

    ###############################################
    ##### normalize latitude and longitude data ###
    ###############################################
    lat_norm = norm_data(lat)
    lon_norm = norm_data(lon)

    ###############################################
    ##### generate final pos data #################
    ###############################################
    pos_data = []
    for j in range(len(lat_norm)):
        pos_data.append([lat_norm[j], lon_norm[j]])

    #############################################
    # saving the pos-beam development dataset for training and validation
    #############################################

    indx = np.arange(1, len(updated_beam) + 1, 1)
    df_new = pd.DataFrame()
    df_new['index'] = indx
    df_new['unit2_pos'] = pos_data
    df_new['unit1_beam_32'] = updated_beam
    df_new['unit1_beam_64'] = original_beam
    df_new.to_csv(fr'./{saved_pos_folder}/scenario{scenario_num}_pos_beam.csv', index=False)

    #############################################
    # generate the train and test dataset
    #############################################
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng), [int(.6 * len(df_new)), int(.9 * len(df_new))])
    train.to_csv(f'./{saved_pos_folder}/scenario{scenario_num}_pos_beam_train.csv', index=False)
    val.to_csv(f'./{saved_pos_folder}/scenario{scenario_num}_pos_beam_val.csv', index=False)
    test.to_csv(f'./{saved_pos_folder}/scenario{scenario_num}_pos_beam_test.csv', index=False)


# 创建图像-位置-波束数据集：
# 结合之前的步骤，同时使用图像数据、位置数据、更新后的波束索引和原始波束索引。
# 创建一个新的DataFrame，分割为训练集、验证集和测试集，并保存为CSV文件。
def create_img_pos_beam_dataset():
    ##folder_to_save = f'./ML_code/{root_folder}/img_pos_beam'
    # folder_to_save = f'./{root_folder}/img_pos_beam'
    # if not os.path.exists(folder_to_save):
    #     os.makedirs(folder_to_save)
    print("进入数据集生成函数")
    
    #############################################
    ###### created updated image path ###########
    #############################################
    updated_img_path = []
    for entry in image_data_lst:
        img_path = entry.split('./')[1]
        updated_path = f'../../../{root_folder}/{img_path}'
        updated_img_path.append(updated_path)

    ###############################################
    ####### read position values from dataset #####
    ###############################################

    lat = []
    lon = []
    ###pos_data_path = df['unit2_loc'].values
    pos_data_path = df['unit1_pwr_1'].values
    for entry in pos_data_path:
        data_to_read = f'./{root_folder}{entry[1:]}'
        pos_val = np.loadtxt(data_to_read)
        # lat_val, lon_val = pos_val[0], pos_val[1]
        lat.append(pos_val[0])
        lon.append(pos_val[1])

    def norm_data(data_lst):
        norm_data = []
        for entry in data_lst:
            norm_data.append((entry - min(data_lst)) / (max(data_lst) - min(data_lst)))
        return norm_data

    ###############################################
    ##### normalize latitude and longitude data ###
    ###############################################
    lat_norm = norm_data(lat)
    lon_norm = norm_data(lon)

    ###############################################
    ##### generate final pos data #################
    ###############################################
    pos_data = []
    for j in range(len(lat_norm)):
        pos_data.append([lat_norm[j], lon_norm[j]])

    #############################################
    # saving the pos-beam development dataset for training and validation
    #############################################

    indx = np.arange(1, len(updated_beam) + 1, 1)
    df_new = pd.DataFrame()
    df_new['index'] = indx
    df_new['unit1_rgb'] = updated_img_path
    df_new['unit2_pos'] = pos_data
    df_new['unit1_beam_32'] = updated_beam
    df_new['unit1_beam_64'] = original_beam
    df_new.to_csv(fr'./{saved_img_pos_folder}/scenario{scenario_num}_pos_beam.csv', index=False)

    #############################################
    # generate the train and test dataset
    #############################################
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng), [int(.6 * len(df_new)), int(.9 * len(df_new))])
    train.to_csv(f'./{saved_img_pos_folder}/scenario{scenario_num}_img_pos_beam_train.csv', index=False)
    val.to_csv(f'./{saved_img_pos_folder}/scenario{scenario_num}_img_pos_beam_val.csv', index=False)
    test.to_csv(f'./{saved_img_pos_folder}/scenario{scenario_num}_img_pos_beam_test.csv', index=False)
    print(f"Image paths: {len(updated_img_path)}")
    print(f"Position data: {len(pos_data)}")
    print(f"Updated beam: {len(updated_beam)}")
    print(f"Original beam: {len(original_beam)}")
    print(f"Index range: {len(indx)}")
    print("生成结束")

# 主函数：
# 依次调用函数以生成和保存所有需要的数据集
if __name__ == "__main__":
    # create_img_beam_dataset()
    # create_pos_beam_dataset()
    create_img_pos_beam_dataset()


