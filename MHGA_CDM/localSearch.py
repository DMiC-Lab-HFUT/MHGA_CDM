import copy
import math
import random

import numpy as np

from tools import *
from queue import Queue
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.metrics import roc_auc_score

def local_search_add_knowledge_test(data, individual, q_matrix, n_students, n_knowledge, n_questions, knowledge_new,
                                    sample_knowledge):
    Aarray = []
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions
    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, sample_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, sample_knowledge)
        Aarray.append(Ai)

    A = [j for i in Aarray for j in i]

    individual = updateIndividual_A_initial(individual, A, n_students, n_knowledge, knowledge_new, sample_knowledge)

    return individual

def local_search_add_knowledge(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                 len_s_g, knowledge_new, sample_knowledge):
    Aarray = []
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions
    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, sample_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, sample_knowledge)
        Aarray.append(Ai)

    A = [j for i in Aarray for j in i]

    slip, guess = ESVESISlipAndGuess(fileterFromQT, filterFromQF, n_students, n_questions)

    individual = updateIndividual_A_initial(individual, A, n_students, n_knowledge, knowledge_new, sample_knowledge)

    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)
    return individual

def local_search_initial_knowledge_test(data, ind, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                         len_s_g, knowledge_new, sample_knowledge,slip, guess):
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, sample_knowledge, n_students)
    individual = updateIndividual_A_initial(ind, A, n_students,n_knowledge, knowledge_new, sample_knowledge)
    return individual


def local_search_initial_knowledge(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                 len_s_g, knowledge_new, sample_knowledge):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)

    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, sample_knowledge, n_students, )

    slip, guess = iRUpdateSandG(A, data, q_matrix,  sample_knowledge, n_students, n_questions)

    individual = updateIndividual_A_initial(individual, A, n_students,n_knowledge, knowledge_new, sample_knowledge)

    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)
    return individual


def local_search_train_EM_update(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                 len_s_g):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)

    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)

    slip, guess = iRUpdateSandG(A, data, q_matrix, n_knowledge, n_students, n_questions)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)

    # TODO：并行可能会出错
    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)
    return individual


def iRUpdateSandG(A, data, q_matrix, n_knowledge, n_students, n_questions):
    A = np.array(A)  # 转化为二维数组
    A = A.reshape(n_students, n_knowledge)
    A = A.tolist()  # 二维列表
    YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    data = data.tolist()
    # IR中的 0 1 2 3  分别表示 IO RO I1 R1
    IR = np.zeros((4, n_questions))  # 4行20列
    IR = IR.tolist()
    for j in range(n_questions):
        for i in range(n_students):
            if YITA[i][j] == 0:
                IR[0][j] = IR[0][j] + 1
            if YITA[i][j] == 0 and data[i][j] == 1:
                IR[1][j] = IR[1][j] + 1
            if YITA[i][j] == 1:
                IR[2][j] = IR[2][j] + 1
            if YITA[i][j] == 1 and data[i][j] == 1:
                IR[3][j] = IR[3][j] + 1
    IR = np.array(IR)
    guess = IR[1] / IR[0]  # 更新g猜测率
    slip = (IR[2] - IR[3]) / IR[2]  # 更新s失误率
    for i in range(n_questions):
        # if slip[i] > threshold_slip:
        #     slip[i] = threshold_slip
        if slip[i] == 0:
            slip[i] = 0.01
        # if guess[i] > threshold_guess:
        #     guess[i] = threshold_guess
        if guess[i] == 0:
            guess[i] = 0.01

    return slip, guess


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_train_DINA(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对
    slip, guess = MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行
    # 个体中更新学生的知识掌握情况
    # TODO：并行可能会出错
    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体
    return individual  # 返回更新后的个体


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_train_RG(data, individual, q_matrix, n_students, n_knowledge_coarse, n_knowledge_fine, n_questions,
                          GENE_LENGTH, len_s_g, data_name):
    slip, guess = acquireSandG(n_students, individual, n_knowledge_fine, GENE_LENGTH, len_s_g)
    #
    # # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
    # YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    # TODO: 应单独拧出来，允许反复创建
    if data_name != 'Math_DMiC':
        groups, groups_2 = random_group(n_knowledge_coarse, n_knowledge_fine)  # 609行 返回粗粒度与细粒度的对应关系
    else:
        groups, groups_2 = read_knowledge_group_from_data(n_knowledge_coarse,
                                                          n_knowledge_fine)  # 577行 读取Math_DMiC 问题的粒度
    # TODO: q_matrix应根据groups进行处理（合并）
    q_coarse = covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions)  # 如果该组内有知识点被考察，则粗粒度
    # 知识点则被考察
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_coarse, n_knowledge_coarse, n_students)
    slip, guess = MStep(IL, r_matrix, data, n_knowledge_coarse, n_students, n_questions)

    individual = updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups)
    # if not math.isnan(slip[0]) or not math.isnan(guess[0]):  h这里会出现 slip[0]为nan然后进入的情况造成异常
    if not math.isnan(slip[0]) and not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge_fine, GENE_LENGTH, len_s_g)
    return individual, groups_2


def local_search_train_IDINA(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)

    # # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量

    A, IL, k_matrix, r_matrix = IEStep(slip, guess, data, q_matrix, n_knowledge, n_students)
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对
    # slip, guess = IMStep(IL, r_matrix, data, n_knowledge, n_students, n_questions)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行
    # 个体中更新学生的知识掌握情况
    # TODO：并行可能会出错
    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体
    return individual  # 返回更新后的个体


def covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions):
    q_coarse = np.zeros((n_questions, n_knowledge_coarse))
    for i in range(n_questions):
        for j in range(n_knowledge_coarse):
            group = groups[j]
            if sum([q_matrix[i][val] for val in group]) >= 1:
                q_coarse[i][j] = 1
            else:
                q_coarse[i][j] = 0

    return q_coarse  # 返回问题粒度


def read_knowledge_group_from_data(n_knowledge_coarse, n_knowledge_fine):
    # 读取 Math_DMiC 集中
    # if granularity >= 3 or granularity < 0:
    #     print("Error!")
    if n_knowledge_coarse == n_knowledge_fine:  # 知识点数
        granularity = 0
    elif n_knowledge_fine == 27:
        granularity = 1
    elif n_knowledge_fine == 170:
        granularity = 2
    else:
        print("Error!")
    file_path = "dataSets/Math_DMiC/q/q_groups_"  # .csv"
    file_path = file_path + str(granularity) + ".csv"
    groups = {}
    groups_2 = {}

    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        data = []
        temp = []
        num_line = 0
        # 取试题对应哪些知识点
        for line in reader:
            line_split = line[0].split(";")  # ;为分隔符
            line_split = [int(each) for each in line_split]
            for each in line_split:
                groups.setdefault(num_line, []).append(each - 1)  # 查找值为num_line
                groups_2[each] = num_line
            temp.append(line_split)
            num_line = num_line + 1
    # groups = {0:[0,1,2,3,4,5,6],1:[7,8,9,10]....}
    # groups_2 ={1:0,2:0,3:0,4:0,5:0,6:0,7:1,...}
    return groups, groups_2


def random_group(n_knowledge_coarse, n_knowledge_fine):
    if n_knowledge_coarse > n_knowledge_fine:
        print('Error: n_knowledge_coarse > n_knowledge_fine')
    groups = {}
    groups_2 = {}

    size_group = math.ceil(n_knowledge_fine / n_knowledge_coarse) - 1  # cell函数，返回数值上的整数
    # 在知识层次未知的情况下，将细粒度的知识点随机整合为一个“粗”粒度的知识点
    num_selected = np.zeros(n_knowledge_coarse)
    max_num_selected = np.zeros(n_knowledge_coarse)
    for i in range(n_knowledge_coarse - 1):
        max_num_selected[i] = size_group
    max_last_idx = n_knowledge_fine - size_group * (n_knowledge_coarse - 1)
    max_num_selected[n_knowledge_coarse - 1] = max_last_idx
    for i in range(n_knowledge_fine):
        id_rnd = random.randint(1, n_knowledge_coarse) - 1  # 生成在1~n_knowledge_coarse随机一个数
        while num_selected[id_rnd] >= max_num_selected[id_rnd]:
            id_rnd = (id_rnd + 1) % n_knowledge_coarse
        num_selected[id_rnd] = num_selected[id_rnd] + 1
        groups.setdefault(id_rnd, []).append(i)
        groups_2[i] = id_rnd

    return groups, groups_2  # 返回粗粒度与细粒度的对应关系


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_test_DINA(data, individual, q_matrix, n_students, n_knowledge, slip, guess):
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)
    return individual


# 针对
# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_test(data, individual, q_matrix, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, slip,
                      guess, data_name):
    # TODO: 应单独拧出来，允许反复创建
    if data_name != 'math_DMiC':
        groups, groups_2 = random_group(n_knowledge_coarse, n_knowledge_fine)
    else:
        groups, groups_2 = read_knowledge_group_from_data(n_knowledge_coarse, n_knowledge_fine)
    # TODO: q_matrix应根据groups进行处理（合并）
    q_coarse = covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions)
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_coarse, n_knowledge_coarse, n_students)
    individual = updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups)

    return individual


def updateIndividual_A_0(individual, A, n_students, n_knowledge):
    individual[0:n_students * n_knowledge] = A
    return individual

def updateIndividual_A_initial(individual, A, n_students, n_knowledge,knowledge_new,sample_knowledge):

    count = 0
    for i in range(n_students):
        for k in range(sample_knowledge):
            individual[i * n_knowledge + knowledge_new[k]] = A[count]
            count += 1
    return individual

def updateIndividual_A_by_question(individual, A):

    for i in range(len(A)):
            if A[i] != 999:
                individual[i] = A[i]
    return individual


def updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups):
    # individual[0:n_students * n_knowledge] = A  # GENE_LENGTH = student * knowlege + question * 2 * len_s_g
    # TODO: test
    for j in range(n_students):
        for i in range(n_knowledge_coarse):
            for idx in groups[i]:
                individual[j * n_knowledge_fine + idx] = A[j * n_knowledge_coarse + i]
    return individual


def updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH, len_s_g):
    # 共question个i, GENE_LENGTH = student * knowledge + question * 2 * len_s_g
    loop = 0
    for i in range(n_knowledge * n_students, GENE_LENGTH, len_s_g * 2):
        # TODO: 是否可以采用混合编码的方式？
        # 这里可能slip中也含有nan
        if not math.isnan(slip[loop]) and not math.isnan(guess[loop]):  # 检查s，g为数值，decode 解码过程
            individual[i:i + len_s_g] = decode_slip(slip[loop], len_s_g)  # 从n_knowledge开始，s为len_s_g的精度，调用 func的231行
            individual[i + len_s_g:i + (len_s_g * 2)] = decode_guess(guess[loop], len_s_g)
        loop = loop + 1

    return individual  # 每个个体的染色体已完成计算。


'''
输入：
IL:学生数*知识点模式数目（2*k)、
sg：分为为失误率（slip）、猜测率（guess）的当前值
n：学生-试题答题情况(Xij)
Q:试题-知识点关联情况
r：r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
k：知识点数目
输出：计算IL
'''


def EStep(slip, guess, data, q_matrix, n_knowledge, n_students):
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)  # 知识考察矩阵
    # crate K matrix，indict k skill could get how many vector
    # 构造K矩阵，表示k个技能可以组成的技能模式矩阵
    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))  # k行 8行 、 2**k列 256列
    for j in range(2 ** n_knowledge):  # 0到255 共256个数
        l = list(bin(j).replace('0b', ''))  # 范围是0到255的二进制 0000 0000-1111 1111 ，l为256种模式
        for i in range(len(l)):  # 0到7 l的长度为8
            k_matrix[n_knowledge - len(l) + i, j] = l[i]  # 8行 256列 的所有可能的技能矩阵赋值完毕

    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    std = np.sum(q_matrix, axis=1)  # 将每一行的元素相加,将Q矩阵压缩为一列 20行1列,每个题目有多少个知识点的阵

    r_matrix = (q_matrix * k_matrix == std) * 1  # Q:20x8 K:8x256  r:20x256  q_matrix为知识考察矩阵，K为8个知识点技能模式矩阵

    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    IL = np.zeros((n_students, 2 ** n_knowledge))  # 行：学生数，列为2^8
    for l in range(2 ** n_knowledge):  # 若k=8: 256列 for循环256次 赋值256次
        # 学生的数量
        lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])  # Xi的边缘似然函数L(Xi|αi)
        IL[:, l] = lll.prod(axis=1)  # prod连乘函数，L（X|α)，当有I个学生时的全体学生边缘化似然函数
    sumIL = IL.sum(axis=1)  # 一行元素相加 428行*1列
    # LX = np.sum([i for i in map(math.log2, sumIL)])  # 似然函数
    # print(LX)
    IL = (IL.T / sumIL).T

    # E-step：现在得到了IL 428行学生*256列知识点模式
    A = []
    for i in range(n_students):
        idx = IL[i].argmax()  # 返回每个学生，argmax 2 ** knowledge 中，最大似然函数的值
        tmp = k_matrix[:, idx].data.tolist()  # k矩阵 8*256 每个题的25
        tmp_array = [i[0] for i in tmp]
        A = A + tmp_array

    return A, IL, k_matrix, r_matrix

    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对


def MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions):
    data = np.array(data)
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对
    # IR中的 0 1 2 3  分别表示 IO RO I1 R1
    IR = np.zeros((4, n_questions))  # 4行20列
    n1 = np.ones((n_students, n_questions))  # 428*20 全1矩阵
    for l in range(2 ** n_knowledge):  # 256次循环
        IR[0] += np.sum(((1 - r_matrix.A[:, l]) * n1).T * IL[:, l], axis=1)  # I0，至少缺乏习题j关联的一个知识点的期望学生数,IL就是p
        IR[1] += np.sum(((1 - r_matrix.A[:, l]) * data).T * IL[:, l], axis=1)  # R0，IO中正确答对第J道题的期望学生数目
        IR[2] += np.sum((r_matrix.A[:, l] * n1).T * IL[:, l], axis=1)  # I1 ，掌握了第j道题所需所有知识点的期望学生数目
        IR[3] += np.sum((r_matrix.A[:, l] * data).T * IL[:, l], axis=1)  # R1 ，I1中正确作答题目J的期望学生数目
    # 针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
    # if (abs(IR[1] / IR[0] - sg[:,1])<threshold).any() and (abs((IR[2]-IR[3]) / IR[2] -sg[:,0])<threshold).any():

    guess = IR[1] / IR[0]  # 更新g猜测率
    slip = (IR[2] - IR[3]) / IR[2]  # 更新s失误率
    # if math.isnan(slip[0]):
    #     print(str(IR[0]))
    for i in range(n_questions):
        if slip[i] > threshold_slip:
            slip[i] = threshold_slip
        if guess[i] > threshold_guess:
            guess[i] = threshold_guess
    # M-step结束 得到IR 0123

    return slip, guess


def IEStep(slip, guess, data, q_matrix, n_knowledge, n_students):
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)  # 知识考察矩阵
    # crate K matrix，indict k skill could get how many vector
    # 构造K矩阵，表示k个技能可以组成的技能模式矩阵
    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))  # k行 8行 、 2**k列 256列
    for j in range(2 ** n_knowledge):  # 0到255 共256个数
        l = list(bin(j).replace('0b', ''))  # 范围是0到255的二进制 0000 0000-1111 1111 ，l为256种模式
        for i in range(len(l)):  # 0到7 l的长度为8
            k_matrix[n_knowledge - len(l) + i, j] = l[i]  # 8行 256列 的所有可能的技能矩阵赋值完毕

    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    std = np.sum(q_matrix, axis=1)  # 将每一行的元素相加,将Q矩阵压缩为一列 20行1列,每个题目有多少个知识点的阵

    r_matrix = (q_matrix * k_matrix == std) * 1  # Q:20x8 K:8x256  r:20x256  q_matrix为知识考察矩阵，K为8个知识点技能模式矩阵
    # r后面保持不变 ，  就是，考察，在学生的每个题目的知识点技能矩阵下，是否满足这道题所考察的知识点，如果满足，则r_matrix为1
    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    IL = np.zeros((n_students, 2 ** n_knowledge))  # 行：学生数，列为2^8
    for l in range(2 ** n_knowledge):  # 若k=8: 256列 for循环256次 赋值256次
        # 学生的数量
        lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])  # Xi的边缘似然函数L(Xi|αi)
        IL[:, l] = lll.prod(axis=1)  # prod连乘函数，L（X|α)，当有I个学生时的全体学生边缘化似然函数
    sumIL = IL.sum(axis=1)  # 一行元素相加 428行*1列
    # LX = np.sum([i for i in map(math.log2, sumIL)])  # 似然函数
    # print(LX)
    IL = (IL.T / sumIL).T

    # E-step：现在得到了IL 428行学生*256列知识点模式
    A = []
    for i in range(n_students):
        idx = IL[i].argmax()  # 每个学生的最大似然函数
        tmp = k_matrix[:, idx].data.tolist()  # k矩阵 8*256 每个题的25
        tmp_array = [i[0] for i in tmp]
        A = A + tmp_array

    return A, IL, k_matrix, r_matrix
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对


def localSearchESVESDTrain(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    #  sj，gj是学生依赖的（SD），它们可以与学生向量αi相关。我们假设sj与αi的掌握技能数（即水平）有关，
    #  这意味着我们认为具有相同技能数的学生在每个问题上都有相同的失误率。此外，我们假设gj与问题j的技能数量缺乏（即不足）有关。
    Aarray = []
    # 从过滤出的
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions

    # 用于DG统计从QF过滤到QT的学生
    SDS = [[0] * n_questions for _ in range(n_knowledge + 1)]
    SDG = [[0] * n_questions for _ in range(n_knowledge + 1)]
    SDDenominator = [[0] * n_questions for _ in range(n_knowledge + 1)]

    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD = ESVECollisionDetectionSD(QT, QF,
                                                                                                             n_knowledge,
                                                                                                             fileterFromQT,
                                                                                                             filterFromQF,
                                                                                                             QTid, QFid)

        QTfromQFforSD = list(set(QTfromQFforSD))
        QFfromQTforSD = list(set(QFfromQTforSD))

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)
        # 第四部：SDS中，加入该学生掌握k个知识点的情况下，哪些题目失误了
        studentI = Ai.count(1)  # 学生i 总共掌握多少个知识点
        if len(QTfromQFforSD) != 0:
            for i in range(0, len(QTfromQFforSD)):
                SDS[studentI][QTfromQFforSD[i]] += 1
        if len(QFfromQTforSD) != 0:
            for i in range(0, len(QFfromQTforSD)):
                SDG[studentI][QFfromQTforSD[i]] += 1

    AMatrix = np.matrix(Aarray)

    # QMatrix = np.matrix(q_matrix)
    # std = np.sum(q_matrix, axis=1)
    # # XMatrix 学生认知掌握情况 和 Q矩阵运算，学生i掌握的知识点是否能答对j题
    # XMatrix = (AMatrix * QMatrix.T == std) * 1
    # XList = XMatrix.tolist()

    # allStudentI 获得每个学生I掌握多少个知识点
    allStudent = AMatrix.sum(axis=1).tolist()

    for i in range(n_students):
        for j in range(n_questions):
            SDDenominator[allStudent[i][0]][j] += 1

    print(SDS)

    A = [j for i in Aarray for j in i]
    SLIP, GUESS = ESVESDSlipAndGuess(Aarray, fileterFromQT, filterFromQF, n_students, n_questions, n_knowledge,
                                     SDS, SDG, SDDenominator)

    slip, guess = ESVESISlipAndGuess(fileterFromQT, filterFromQF, n_students, n_questions)

    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行

    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体

    return individual, SLIP, GUESS


def localSearchESVESITrain(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    Aarray = []
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions
    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)

    A = [j for i in Aarray for j in i]

    slip, guess = ESVESISlipAndGuess(fileterFromQT, filterFromQF, n_students, n_questions)

    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行

    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体
    return individual  # 返回更新后的个体


def localSearchESVETrain(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    Aarray = []
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions
    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)

    A = [j for i in Aarray for j in i]

    slip, guess = iRUpdateSandG(A, data, q_matrix, n_knowledge, n_students, n_questions)

    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行

    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体
    return individual  # 返回更新后的个体


def localSearchESVETest(data, individual, q_matrix, n_students, n_knowledge):
    filterFromQF = [0] * len(q_matrix)
    fileterFromQT = [0] * len(q_matrix)
    n_questions = len(q_matrix)
    Aarray = []
    for i in range(n_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        # 第二步:得到Qt 和 Qf 的可靠矩阵
        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        Ai = ESVEFromQreal(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)

    A = [j for i in Aarray for j in i]

    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)
    return individual


def ESVESISlipAndGuess(fileterFromQT, filterFromQF, n_students, n_questions):
    slip = [0] * n_questions
    guess = [0] * n_questions

    for j in range(n_questions):
        slip[j] = filterFromQF[j] / n_students
        guess[j] = fileterFromQT[j] / n_students

    for i in range(n_questions):
        if slip[i] == 0:
            slip[i] = 0.01
        if guess[i] == 0:
            guess[i] = 0.01

        # if slip[i] > threshold_slip:
        #     slip[i] = threshold_slip
        # if guess[i] > threshold_guess:
        #     guess[i] = threshold_guess



    return slip, guess


def ESVESDSlipAndGuess(Aarray, fileterFromQT, filterFromQF, n_students, n_questions, n_knowledge, SDS, SDG,
                       SDDenominator):
    # 学生的失误率和猜测率和他掌握的K知识点数目有关

    # slip = [[0 for j in range(n_knowledge)] for i in range(n_questions)]
    # guess = [[0 for j in range(n_knowledge)] for i in range(n_questions)]
    SDS = np.matrix(SDS)
    SDG = np.matrix(SDG)
    SDDenominator = np.matrix(SDDenominator)

    slip = SDS / SDDenominator
    guess = SDG / SDDenominator
    print("todo:")

    return slip, guess


def acquireQTandQF(data, q_matrix, n_questions):
    QT = []
    QF = []
    QTid = []
    QFid = []
    for j in range(n_questions):
        if data[j] == 1:
            QT.append(q_matrix[j])
            QTid.append(j)
        else:
            QF.append(q_matrix[j])
            QFid.append(j)
    return QT, QF, QTid, QFid


def ESVECollisionDetectionSD(QT, QF, n_knowledge, fileterFromQT, filterFromQF, QTid, QFid):
    QFfromQTforSD = []
    QTfromQFforSD = []

    fileter = [0] * (len(QT) + len(QF))

    if len(QT) == 0 or len(QF) == 0:
        QTreal = QT
        QFreal = QF
        return QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD

    QTRecurrence = []
    maxCollisionQ = 9999
    while maxCollisionQ != 0:
        if len(QT) == 0 or len(QF) == 0:
            QTreal = QT
            QFreal = QF
            return QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD
        QTConfiltDegrees = [0] * len(QT)
        QFConfiltDegrees = [0] * len(QF)
        for t in range(len(QT)):
            for f in range(len(QF)):
                count = 0
                for k in range(n_knowledge):
                    if QT[t][k] >= QF[f][k]:
                        count += 1
                if count == n_knowledge:
                    QTConfiltDegrees[t] += 1
                    QFConfiltDegrees[f] += 1
        # 获得QF中最大冲突的题目号
        # maxCollision = max(QFConfiltDegrees)
        maxQF = max(QFConfiltDegrees)
        maxQT = max(QTConfiltDegrees)
        QTRecurrence.append(maxQT)
        if len(QTRecurrence) > 2:
            if QTRecurrence[-3] == QTRecurrence[-1]:
                QTreal = QT
                QFreal = QF
                return QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD

        if maxQT == 0 or maxQF == 0:
            QTreal = QT
            QFreal = QF
            return QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD

        if maxQF > maxQT:
            maxCollisionQFIndex = QFConfiltDegrees.index(maxQF)
            fileterid = QFid[maxCollisionQFIndex]
            if fileter[fileterid] == 0:
                filterFromQF[fileterid] += 1
                fileter[fileterid] += 1

            QT.append(QF[maxCollisionQFIndex])
            QTid.append(QFid[maxCollisionQFIndex])
            if QFid[maxCollisionQFIndex] not in QFfromQTforSD:
                QTfromQFforSD.append(QFid[maxCollisionQFIndex])
            del QF[maxCollisionQFIndex]
            del QFid[maxCollisionQFIndex]
            maxCollisionQ = maxQF
        else:
            maxCollisionQTIndex = QTConfiltDegrees.index(maxQT)
            fileterid = QTid[maxCollisionQTIndex]
            if fileter[fileterid] == 0:
                fileterFromQT[fileterid] += 1
                fileter[fileterid] += 1

            QF.append(QT[maxCollisionQTIndex])
            QFid.append(QTid[maxCollisionQTIndex])
            if QTid[maxCollisionQTIndex] not in QTfromQFforSD:
                QFfromQTforSD.append(QTid[maxCollisionQTIndex])
            del QT[maxCollisionQTIndex]
            del QTid[maxCollisionQTIndex]
            maxCollisionQ = maxQF

    QTreal = QT
    QFreal = QF
    # print(QTConfiltDegrees)
    # print(QFConfiltDegrees)
    return QTreal, QFreal, fileterFromQT, filterFromQF, QTfromQFforSD, QFfromQTforSD


def ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT, filterFromQF, QTid, QFid):
    fileter = [0] * (len(QT) + len(QF))

    if len(QT) == 0 or len(QF) == 0:
        QTreal = QT
        QFreal = QF
        return QTreal, QFreal, fileterFromQT, filterFromQF

    QTRecurrence = []
    maxCollisionQ = 9999
    while maxCollisionQ != 0:
        if len(QT) == 0 or len(QF) == 0:
            QTreal = QT
            QFreal = QF
            return QTreal, QFreal, fileterFromQT, filterFromQF
        QTConfiltDegrees = [0] * len(QT)
        QFConfiltDegrees = [0] * len(QF)

        QFSumK = []
        for f in range(len(QF)):
            temNum = 0
            for k in range(n_knowledge):
                if QF[f][k] == 1:
                    temNum += 1
            QFSumK.append(temNum)


        for t in range(len(QT)):
            for f in range(len(QF)):
                count = 0
                for k in range(n_knowledge):
                    #todo: 第一次的思路
                    # if QT[t][k] > QF[f][k]:
                    # QTConfiltDegrees[t] += 1
                    # QFConfiltDegrees[f] += 1

                    # todo : 第二次的思路
                    # if QT[t][k] >= QF[f][k]:
                    #     count += 1
                    # if count == n_knowledge:
                    #     QTConfiltDegrees[t] += 1
                    #     QFConfiltDegrees[f] += 1

                    if QF[f][k] == 1 and QT[t][k] == 1:
                        count += 1
                if count == QFSumK[f]:
                    QTConfiltDegrees[t] += 1
                    QFConfiltDegrees[f] += 1


        # 获得QF中最大冲突的题目号
        # maxCollision = max(QFConfiltDegrees)
        maxQF = max(QFConfiltDegrees)
        maxQT = max(QTConfiltDegrees)
        QTRecurrence.append(maxQT)
        if len(QTRecurrence) > 2:
            if QTRecurrence[-3] == QTRecurrence[-1]:
                QTreal = QT
                QFreal = QF
                return QTreal, QFreal, fileterFromQT, filterFromQF

        if maxQT == 0 or maxQF == 0:
            QTreal = QT
            QFreal = QF
            return QTreal, QFreal, fileterFromQT, filterFromQF

        if maxQF > maxQT:
            maxCollisionQFIndex = QFConfiltDegrees.index(maxQF)
            fileterid = QFid[maxCollisionQFIndex]
            if fileter[fileterid] == 0:
                filterFromQF[fileterid] += 1
                fileter[fileterid] += 1

            QT.append(QF[maxCollisionQFIndex])
            QTid.append(QFid[maxCollisionQFIndex])
            del QF[maxCollisionQFIndex]
            del QFid[maxCollisionQFIndex]
            maxCollisionQ = maxQF
        else:
            maxCollisionQTIndex = QTConfiltDegrees.index(maxQT)
            fileterid = QTid[maxCollisionQTIndex]
            if fileter[fileterid] == 0:
                fileterFromQT[fileterid] += 1
                fileter[fileterid] += 1

            QF.append(QT[maxCollisionQTIndex])
            QFid.append(QTid[maxCollisionQTIndex])
            del QT[maxCollisionQTIndex]
            del QTid[maxCollisionQTIndex]
            maxCollisionQ = maxQF

    QTreal = QT
    QFreal = QF
    # print(QTConfiltDegrees)
    # print(QFConfiltDegrees)
    return QTreal, QFreal, fileterFromQT, filterFromQF


def ESVEFromQreal(QTreal, QFreal, n_knowledge):
    A = [999] * n_knowledge

    QtCount = [0] * n_knowledge
    QfCount = [0] * n_knowledge

    for j in range(len(QTreal)):
        for k in range(n_knowledge):
            if QTreal[j][k] == 1:
                QtCount[k] += 1
    if len(QFreal) != 0:
        for j in range(len(QFreal)):
            for k in range(n_knowledge):
                if QFreal[j][k] == 1:
                    QfCount[k] += 1

    for i in range(n_knowledge):
        if QtCount[i] != 0:
            A[i] = 1

    for i in range(n_knowledge):
        # if QfCount[i] == 1 and A[i] == 999:
        if QfCount[i] >= 1 and A[i] == 999:

            A[i] = 0

    for i in range(n_knowledge):
        if A[i] == 999:
            # A[i] = 1
            if random.random() < 0.5:
                A[i] = 0
            else:
                A[i] = 1

    return A

def ESVEFromQrealByQuestion(QTreal, QFreal, n_knowledge):
    A = [999] * n_knowledge

    QtCount = [0] * n_knowledge
    QfCount = [0] * n_knowledge

    for j in range(len(QTreal)):
        for k in range(n_knowledge):
            if QTreal[j][k] == 1:
                QtCount[k] += 1
    if len(QFreal) != 0:
        for j in range(len(QFreal)):
            for k in range(n_knowledge):
                if QFreal[j][k] == 1:
                    QfCount[k] += 1

    for i in range(n_knowledge):
        if QtCount[i] != 0:
            A[i] = 1

    for i in range(n_knowledge):
        # if QfCount[i] == 1 and A[i] == 999:
        if QfCount[i] >= 1 and A[i] == 999:
            A[i] = 0

    # for i in range(n_knowledge):
    #     if A[i] == 999:
    #         # A[i] = 1
    #         if random.random() < 0.5:
    #             A[i] = 0
    #         else:
    #             A[i] = 1

    return A
