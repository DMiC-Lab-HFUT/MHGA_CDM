import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from myAlgorithms import *
from multiprocessing import Pool


def testModel(testModel_name, tem_files, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data,
              q_matrix, slip, guess,
              data_patch_id, run_id=1,
              n_pop=50, max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):
    multi = False
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1 = 0
    AUC = 0
    MAE = 0
    RMSE = 0

    print(testModel_name)
    Accuracy, Precision, Recall, F1, AUC, MAE, RMSE = testModel_EA(tem_files, n_students, n_questions,
                                                                   n_knowledge_coarse,
                                                                   n_knowledge_fine, len_s_g, data, q_matrix, slip,
                                                                   guess,
                                                                   data_patch_id, run_id, n_pop, max_generations,
                                                                   alg_name,
                                                                   data_name)

    return Accuracy, Precision, Recall, F1, AUC, MAE, RMSE


def testModel_EA(tem_files, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix,
                 slip,
                 guess, data_patch_id, run_id, n_pop, max_generations, alg_name, data_name):
    startTime = time.time()
    flag_train = False
    n_knowledge = n_knowledge_fine
    global_guide_ind = None
    if alg_name == 'IGA':
        resultPop, logbook, slip, guess, global_guide_ind = IGA(tem_files, n_students, n_questions, n_knowledge_coarse,
                                                                n_knowledge_fine, data,
                                                                q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                                flag_train, max_generations, len_s_g, alg_name,
                                                                data_name)


    label = []
    from sklearn.metrics import confusion_matrix
    bestIndividual = list(global_guide_ind)
    bestA = acquireA(bestIndividual, n_students, n_knowledge)
    bestYITA = acquireYITA(bestA, q_matrix, n_students, n_questions, n_knowledge)

    slip = np.array(slip)
    guess = np.array(guess)
    bestX, Xscore = acquireX(n_students, n_questions, bestYITA, slip, guess)  #
    predict = []
    for rrr, nnn in zip(bestX, data):
        predict.extend(rrr)
        label.extend(nnn)
    predictScore = []
    for i in range(n_students):
        for j in range(n_questions):
            predictScore.append(Xscore[i][j])
    C2 = confusion_matrix(label, predict, labels=[0, 1])

    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    try:
        AUC = roc_auc_score(label, predict)
    except ValueError:
        AUC = 0.5
    MAE = metrics.mean_absolute_error(label, predict)
    RMSE = metrics.mean_squared_error(label, predict) ** 0.5

    print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    print('------------------预测结束-----------------------------')
    return Accuracy, Precision, Recall, F1, AUC, MAE, RMSE

