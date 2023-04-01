import os
import random
# myAlgorithms.py  引用该库
import time

import pandas as pd
from func import *
import datetime


def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)  # index=False 避免生成行索引


def data_reader(data_name):
    # 试题-知识点矩阵
    q_path = "dataSets/" + data_name + "/q.csv"

    # 学生-答题矩阵
    data_path = "dataSets/" + data_name + "/data.csv"

    # q_str = "q" + str(granularity)
    # q_path_pre = "dataSets\\" + data_name + "\\q\\" + q_str + "\\" + q_str + "_" + str(ith)
    # q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:  # 判断括号里的文件是否存在的意思
        q_path_pre = "dataSets/" + data_name + "/q"
        txt2csv(q_path_pre)  # 将txt文档转换为csv文档

    if os.path.exists(data_path) == False:
        data_path_pre = "dataSets/" + data_name + "/data"
        txt2csv(data_path_pre)

    q_matrix = acquireQ(q_path)
    data = acquireData(data_path)
    # data = random.sample(data, 100)
    data = np.array(data)  #

    return q_matrix, data


def data_reader_math_DMiC(data_name, ith=1):
    granularity = 0
    # 试题-知识点矩阵
    q_str = "q" + str(granularity)
    q_path_pre = "dataSets\\" + data_name + "\\q\\" + q_str + "\\" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_0 = acquireQ(q_path)  # 二维列表 15*11   读取试题*知识点考察矩阵

    granularity = 1
    # 试题-知识点矩阵
    q_str = "q" + str(granularity)
    q_path_pre = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_1 = acquireQ(q_path)  # 二维列表 15*11

    granularity = 2
    # 试题-知识点矩阵
    q_str = "q" + str(granularity)
    q_path_pre = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_2 = acquireQ(q_path)  # 二维列表 15*11

    # 学生-答题矩阵
    data_path_pre = "dataSets/" + data_name + "/data/data" + str(ith)
    data_path = data_path_pre + ".csv"
    if os.path.exists(data_path) == False:
        txt2csv(data_path_pre)
    data = acquireData(data_path)  # 读取学生实际的答题情况

    data = np.array(data)  # 二维数组 40学生*15题目

    return q_matrix_0, q_matrix_1, q_matrix_2, data


def q_matrix_reader_math_DMiC(data_name, ith=1):
    granularity = 0
    # 试题-知识点矩阵
    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_0 = acquireQ(q_path)  # 二维列表 15*11

    granularity = 1
    # 试题-知识点矩阵
    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_1 = acquireQ(q_path)  # 二维列表 15*11

    granularity = 2
    # 试题-知识点矩阵
    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_2 = acquireQ(q_path)  # 二维列表 15*11

    return q_matrix_0, q_matrix_1, q_matrix_2


def files_create(alg, data_name):
    dt = time.localtime()
    time_String = time.strftime("%Y-%m-%d-%H_%M_%S", dt)
    tem_file_name = "Log/" + time_String + "_" + data_name + "_" + alg
    tem_file_name_pict_train = "Log/" + time_String + "_" + data_name + "_" + alg + "/" + "train"
    tem_file_name_pict_test = "Log/" + time_String + "_" + data_name + "_" + alg + "/" + "test"
    if not os.path.exists(tem_file_name):
        os.makedirs(tem_file_name)
        os.makedirs(tem_file_name_pict_train)
        os.makedirs(tem_file_name_pict_test)
    return tem_file_name


def files_open(tem_files, alg, data_name, data_patch_i, runID):
    filename = tem_files + "/" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    filename = tem_files + "/" + "_score_" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record_data = open(filename, 'a')
    return f_record, f_record_data


def files_close(f_record, f_record_data):
    f_record.close()
    f_record_data.close()


def save_final_results(tem_files, results_per_run, data_patch_id, mean_accuracy, mean_precision, mean_recall, mean_f1,
                       mean_auc,mean_MAE, mean_RMSE,max_runs, alg_name,data_name):
    filename = tem_files + "/" + "_" + str(data_patch_id) + "_max_run" + str(
        max_runs) + "_.txt"
    print(filename)
    str_results = str(mean_accuracy) + "    " + str(mean_precision) + "  " + str(mean_recall) + "   " + str(
        mean_f1) + "       " + str(mean_auc) + "     " + str(mean_MAE) + "       " + str(mean_RMSE)
    str_index = " mean_accuracy       mean_precision       mean_recall      mean_f1         mean_auc         " \
                "mean_mae         mean_rmse"
    print("accuracy, precision, recall, f1, auc, mae, rmse  " + str_results)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_results + '\n')
    str_index_1 = " run_id   mean_accuracy        mean_precision       mean_recall      mean_f1         mean_auc  " \
                  "mean_mae         mean_rmse"
    f_record.writelines(str_index_1 + '\n')
    for i in range(max_runs):
        # (run_id, accuracy, precision, recall, f1, auc)
        f_record.writelines(str(results_per_run[i]) + '\n')

    f_record.close()


def save_final_results_average(tem_files, average_accuracy, average_precision, average_recall, average_f1, average_auc,
                               average_MAE, average_RMSE, max_runs,alg_name, data_name):  # 三次大循环，5折的总平均值
    filename = tem_files + "/" + "final_average_max_run ：" + str(max_runs) + "_.txt"
    print(filename)
    str_average = str(average_accuracy) + "      " + str(average_precision) + "      " + str(
        average_recall) + "     " + str(average_f1) + "       " + str(average_auc)  + "       " + str(average_MAE) \
                  + "       " + str(average_RMSE)
    str_index = " mean_accuracy           mean_precision               mean_recall            mean_f1        " \
                "      mean_auc     mean_mae         mean_rmse"
    print("accuracy, precision, recall, f1, auc, mae, rmse " + str_average)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_average + '\n')
    f_record.close()


def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)  # index=False 避免生成行索引


# 以下四个函数是用来写入 S,G,A,X等信息

def Write_sg(slip, guess, data_patch_i, runID):
    filename = "Log/reader/sg_" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    f_record.write("每个题的失误率slip：\n")
    for i in range(len(slip)):
        f_record.write(str(i) + ": " + str(slip[i]) + '   ')
    f_record.write("\n每个题的猜测率slip：\n")
    for j in range(len(guess)):
        f_record.write(str(j) + ": " + str(guess[j]) + '  ')
    f_record.close()


def Write_A(A, data_patch_i, runID):
    a = np.array(A)
    Ai, Aj = a.shape
    filename = "Log/reader/A_" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(Ai):
        f_record.write("id" + str(i) + ":  ")
        for j in range(Aj):
            f_record.write(str(A[i][j]) + '  ')
        f_record.write('\n')
    f_record.close()


def Write_X(X, data_patch_i, runID):
    x = np.array(X)
    xi, xj = x.shape
    filename = "Log/reader/X_" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(xi):
        f_record.write("id" + str(i) + ":  ")
        for j in range(xj):
            f_record.write(str(X[i][j]) + '  ')
        f_record.write('\n')
    f_record.close()


def Write_rrrSum(rrrSum, data_patch_i, runID):
    Qi = 300
    Qj = 10
    filename = "Log/reader/rrrSum_" + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(Qi):
        f_record.write("id" + str(i) + ":  ")
        for j in range(Qj):
            f_record.write(str(rrrSum[i * 10 + j]) + '  ')
        f_record.write('\n')
    f_record.close()
