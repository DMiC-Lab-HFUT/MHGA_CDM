from train_model import *
from testModel import *
from multiprocessing import Pool
from sklearn.model_selection import KFold, train_test_split
import warnings

warnings.filterwarnings("ignore")  # 忽略正常运行时的错误


def single_run(testModel_name, tem_files, data_patch_id, run_id, num_train_students, n_questions, n_knowledge_coarse,
               n_knowledge_fine,
               len_s_g,
               data_train, q_matrix, data_test, num_test_students, alg_name, data_name, max_generations, n_pop):
    random.seed(run_id)

    f_record, f_record_data = files_open(tem_files, alg_name, data_name, data_patch_id, run_id)
    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次训练")
    slip, guess = trainModel(tem_files, num_train_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g,
                             data_train,
                             q_matrix, data_patch_id, run_id, n_pop, max_generations, alg_name, data_name)

    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次测试")
    accuracy, precision, recall, f1, auc, MAE, RMSE = testModel(testModel_name, tem_files, num_test_students,
                                                                n_questions, n_knowledge_coarse,
                                                                n_knowledge_fine, len_s_g, data_test,
                                                                q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                                max_generations, alg_name,
                                                                data_name)  # testModel.py  6行
    f_record.writelines("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次" + "\n" + "Accuracy:" + str(accuracy) +
                        "\t" + "Precision:" + str(precision) + "\t" + "Recall:" + str(recall) + "\t" + "F1:" + str(f1) +
                        "\t" + "AUC:" + str(auc) + "\t" + "MAE" + str(MAE) + "\t" + "RMSE" + str(RMSE) + "\t" + "\n")
    f_record_data.writelines(str(data_patch_id) + "." + str(run_id) + "\t" + str(accuracy) + "\t" +
                             str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\t" + str(auc) + "\t" + "MAE" +
                             str(MAE) + "\t" + "RMSE" + str(RMSE) + "\t" + "\n")
    files_close(f_record, f_record_data)
    return accuracy, precision, recall, f1, auc, MAE, RMSE, run_id, tem_files


def prepare(dataset_idx=0, granularity=0):
    # 数据读取初始化
    data_name = data_names[dataset_idx]

    if data_name == 'Math_DMiC':
        q_matrix_0, q_matrix_1, q_matrix_2, data = data_reader_math_DMiC(data_name)
        # data_reader_math_DMiC 位于tools.py文件
        if granularity == 0:
            q_matrix = q_matrix_0
        elif granularity == 1:
            q_matrix = q_matrix_1
        elif granularity == 2:
            q_matrix = q_matrix_2
        q_matrix_coarse = q_matrix_0  # 粗粒度
    else:
        q_matrix, data = data_reader(data_name)
        q_matrix_coarse = q_matrix
    print(len(q_matrix[0]))  # 170个知识点
    return data, q_matrix, q_matrix_coarse


def run_main(testModel_name, tem_files, alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations,
             n_pop, multi, n_knowledge_coarse):

    KF = KFold(n_splits=max_split, shuffle=False)
    data_patch_i = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = 0
    for train_index, test_index in KF.split(data):
        data_patch_i = data_patch_i + 1

        data_train, data_test = data[train_index], data[test_index]
        num_train_students = len(data_train)
        num_test_students = len(data_test)
        n_questions = len(q_matrix)
        n_knowledge = len(q_matrix[0])
        len_s_g = 15
        n_knowledge_fine = n_knowledge
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_f1 = 0
        mean_auc = 0
        mean_MAE = 0
        mean_RMSE = 0
        array_accuracy = []
        array_precision = []
        array_recall = []
        array_f1 = []
        array_auc = []
        array_MAE = []
        array_RMSE = []
        results_per_run = []

        if multi:
            print('multi：' + str(max_runs) + ' processes')
            pool = Pool(processes=max_runs)

            multiple_results = []
            for run_id in range(max_runs):
                multiple_results.append(
                    pool.apply_async(single_run,
                                     (testModel_name, tem_files, data_patch_i, run_id, num_train_students, n_questions,
                                      n_knowledge_coarse, n_knowledge_fine, len_s_g, data_train,
                                      q_matrix, data_test, num_test_students, alg_name,
                                      data_name, max_generations, n_pop)))

            pool.close()
            pool.join()

            for res in multiple_results:
                accuracy, precision, recall, f1, auc, MAE, RMSE, run_id, tem_files = res.get()
                mean_accuracy += accuracy
                mean_precision += precision
                mean_recall += recall
                mean_f1 += f1
                mean_auc += auc
                mean_MAE += MAE
                mean_RMSE += RMSE
                array_precision.append(precision)
                array_recall.append(recall)
                array_f1.append(f1)
                array_auc.append(auc)
                array_MAE.append(MAE)
                array_RMSE.append(RMSE)
                results_per_run.append((run_id, accuracy, precision, recall, f1, auc, MAE, RMSE))
        else:
            for run_id in range(max_runs):
                accuracy, precision, recall, f1, auc, MAE, RMSE, run_id, tem_files = single_run(testModel_name,
                                                                                                tem_files,
                                                                                                data_patch_i, run_id,
                                                                                                num_train_students,
                                                                                                n_questions,
                                                                                                n_knowledge_coarse,
                                                                                                n_knowledge_fine,
                                                                                                len_s_g,
                                                                                                data_train,
                                                                                                q_matrix, data_test,
                                                                                                num_test_students,
                                                                                                alg_name,
                                                                                                data_name,
                                                                                                max_generations, n_pop)
                mean_accuracy += accuracy
                mean_precision += precision
                mean_recall += recall
                mean_f1 += f1
                mean_auc += auc
                mean_MAE += MAE
                mean_RMSE += RMSE
                array_accuracy.append(accuracy)
                array_precision.append(precision)
                array_recall.append(recall)
                array_f1.append(f1)
                array_auc.append(auc)
                array_MAE.append(MAE)
                array_RMSE.append(RMSE)
                results_per_run.append((run_id, accuracy, precision, recall, f1, auc, MAE, RMSE))  # 每次的运行结果
        mean_accuracy /= max_runs
        mean_precision /= max_runs
        mean_recall /= max_runs
        mean_f1 /= max_runs
        mean_auc /= max_runs
        mean_MAE /= max_runs
        mean_RMSE /= max_runs
        save_final_results(tem_files, results_per_run, data_patch_i, mean_accuracy, mean_precision, mean_recall,
                           mean_f1, mean_auc, mean_MAE, mean_RMSE, max_runs, alg_name, data_name)
        s1 += mean_accuracy
        s2 += mean_precision
        s3 += mean_recall
        s4 += mean_f1
        s5 += mean_auc
        s6 += mean_MAE
        s7 += mean_RMSE
    average_accuracy = s1 / max_split
    average_precision = s2 / max_split
    average_recall = s3 / max_split
    average_f1 = s4 / max_split
    average_auc = s5 / max_split
    average_MAE = s6 / max_split
    average_RMSE = s7 / max_split
    save_final_results_average(tem_files, average_accuracy, average_precision, average_recall, average_f1, average_auc,
                               average_MAE, average_RMSE, max_runs, alg_name, data_name)


if __name__ == '__main__':
    sum_ALS = 0
    startTimeA = time.time()
    max_runs = 1
    max_split = 5
    is_multi = False
    max_generations = 100
    n_pop = 50
    alg_names = ["IGA"]
    data_names = ["FrcSub", "Math1", "Math2"]
    testModel_names = ["testModel_EA"]
    alg_id = 0
    dataset_id = 0
    testModel_name_id = 0
    granularity = 0
    NumTime = 0
    for alg_id in range(0, 1):
        for dataset_id in range(0, 1):
            for testModel_name_id in range(0, 1):
                NumTime = NumTime + 1
                testModel_name = testModel_names[testModel_name_id]
                alg_name = alg_names[alg_id]
                data_name = data_names[dataset_id]
                print_info = str(NumTime) + "##" + str(alg_id) + " " + alg_name + " " + str(
                    dataset_id) + " " + data_name
                tem_files = files_create(alg_name, data_name)
                print(print_info)
                data, q_matrix, q_matrix_coarse = prepare(dataset_id, granularity)

                n_questions, n_knowledge_coarse = np.mat(q_matrix).shape
                print(n_knowledge_coarse)
                run_main(testModel_name, tem_files, alg_name, data_name, data, q_matrix, max_runs, max_split,
                         max_generations, n_pop,
                         is_multi, n_knowledge_coarse)
                with open(tem_files + "/" + 'ALS.txt', 'a+') as f:
                    reader = f.readlines()
                    for row in reader:

                        sum_ALS = sum_ALS + int(row)

    print('总用时：' + str(time.time() - startTimeA) + '秒')
