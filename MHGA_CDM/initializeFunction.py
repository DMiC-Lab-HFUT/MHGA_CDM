import random
from localSearch import *


# 初始化函数，使用EM算法的一两步更新 2/3 的染色体，保持 1/3染色体的随机性
def initial_population(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH, len_s_g,
                       sample_knowledge):
    # 1. 从知识点中随机抽取2/3的知识点，作为EM算法要更新的知识点
    knowledge_list = [i for i in range(n_knowledge)]
    # sample_knowledge = int((n_knowledge / 3) * 2)
    knowledge_new = random.sample(knowledge_list, sample_knowledge)
    knowledge_new.sort()
    # print(knowledge_new)
    q_matrix_new = []
    for j in range(n_questions):
        q_jth = []
        for k in range(len(knowledge_new)):
            q_jth.append(q_matrix[j][knowledge_new[k]])
        q_matrix_new.append(q_jth)
    # print(q_matrix_new)

    # 2. 利用EM算法初始化学生掌握情况

    # for ind in pop:
    #     ind = local_search_knowledge(data, ind, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
    #                                  len_s_g)
    #     ind.fitness.values = evaluate(ind)
    #     print(ind.fitness.values)

    ind = local_search_initial_knowledge(data, ind, q_matrix_new, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                         len_s_g, knowledge_new, sample_knowledge)
    return ind


def add_population(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH, len_s_g,
                   sample_knowledge):
    # 1. 从知识点中随机抽取2/3的知识点，作为EM算法要更新的知识点
    knowledge_list = [i for i in range(n_knowledge)]
    # sample_knowledge = int((n_knowledge / 3) * 2)
    knowledge_new = random.sample(knowledge_list, sample_knowledge)
    knowledge_new.sort()
    # print(knowledge_new)
    q_matrix_new = []
    for j in range(n_questions):
        q_jth = []
        for k in range(len(knowledge_new)):
            q_jth.append(q_matrix[j][knowledge_new[k]])
        q_matrix_new.append(q_jth)
    # print(q_matrix_new)

    # 2. 利用ESVE算法初始化学生掌握情况
    ind = local_search_add_knowledge(data, ind, q_matrix_new, n_students, n_knowledge, n_questions, GENE_LENGTH,
                                     len_s_g, knowledge_new, sample_knowledge)
    return ind


def initial_population_test(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH, len_s_g,
                            sample_knowledge, slip, guess):
    # 1. 从知识点中随机抽取2/3的知识点，作为EM算法要更新的知识点
    knowledge_list = [i for i in range(n_knowledge)]
    # sample_knowledge = int((n_knowledge / 3) * 2)
    knowledge_new = random.sample(knowledge_list, sample_knowledge)
    knowledge_new.sort()
    # print(knowledge_new)
    q_matrix_new = []
    for j in range(n_questions):
        q_jth = []
        for k in range(len(knowledge_new)):
            q_jth.append(q_matrix[j][knowledge_new[k]])
        q_matrix_new.append(q_jth)
    # print(q_matrix_new)

    # 2. 利用EM算法初始化学生掌握情况

    # for ind in pop:
    #     ind = local_search_knowledge(data, ind, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
    #                                  len_s_g)
    #     ind.fitness.values = evaluate(ind)
    #     print(ind.fitness.values)

    ind = local_search_initial_knowledge_test(data, ind, q_matrix_new, n_students, n_knowledge, n_questions,
                                              GENE_LENGTH,
                                              len_s_g, knowledge_new, sample_knowledge, slip, guess)
    return ind


def add_population_test(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH, len_s_g,
                        sample_knowledge, slip, guess):
    # 1. 从知识点中随机抽取2/3的知识点，作为EM算法要更新的知识点
    knowledge_list = [i for i in range(n_knowledge)]
    # sample_knowledge = int((n_knowledge / 3) * 2)
    knowledge_new = random.sample(knowledge_list, sample_knowledge)
    knowledge_new.sort()
    # print(knowledge_new)
    q_matrix_new = []
    for j in range(n_questions):
        q_jth = []
        for k in range(len(knowledge_new)):
            q_jth.append(q_matrix[j][knowledge_new[k]])
        q_matrix_new.append(q_jth)
    # print(q_matrix_new)

    # 2. 利用ESVE算法初始化学生掌握情况
    ind = local_search_add_knowledge_test(data, ind, q_matrix_new, n_students, n_knowledge, n_questions,
                                          knowledge_new, sample_knowledge)
    return ind


def local_search_train_ESVE_by_students(data, data_all, data_ESVE_id, seed_LS, seed_LS_tem, q_matrix, n_students,
                                        n_questions, new_students, flag_train, GENE_LENGTH, len_s_g, slip, guess):
    n_knowledge = len(q_matrix[0])
    Aarray = []
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions

    for i in range(new_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data_all[i], q_matrix, n_questions)

        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        # ESVEFromQrealByQuestion
        Ai = ESVEFromQrealByQuestion(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)

    Seed_LS_A = seed_LS[:n_students * len(q_matrix[0])]
    for i in range(new_students):
        for j in range(len(q_matrix[0])):
            if Aarray[i][j] != 999:
                Seed_LS_A[data_ESVE_id[i] * len(q_matrix[0]) + j] = Aarray[i][j]
    individual = updateIndividual_A_0(seed_LS, Seed_LS_A, n_students, n_knowledge)

    second_LS_id = []
    second_LS = []

    seed_LS_X = acquireXByInd(seed_LS, q_matrix, n_students, n_questions, n_knowledge, slip, guess, flag_train,
                              GENE_LENGTH, len_s_g)
    seed_LS_tem_X = acquireXByInd(seed_LS_tem, q_matrix, n_students, n_questions, n_knowledge, slip, guess, flag_train,
                                  GENE_LENGTH, len_s_g)

    for i in range(new_students):
        sum_tem = 0
        sum_seed = 0
        for j in range(n_questions):
            if seed_LS_X[i][j] == data[data_ESVE_id[i]][j]:
                sum_seed += 1
            if seed_LS_tem_X[i][j] == data[data_ESVE_id[i]][j]:
                sum_tem += 1
        if sum_seed < sum_tem:
            seed_LS[data_ESVE_id[i] * n_knowledge: (data_ESVE_id[i] + 1) * n_knowledge] = \
                seed_LS_tem[data_ESVE_id[i] * n_knowledge: (data_ESVE_id[i] + 1) * n_knowledge]
            second_LS_id.append(data_ESVE_id[i])
            second_LS.append(data_all[i])

    # if len(second_LS_id) != 0:
    #     print("无效的学生数目为" + str(len(second_LS_id)))

    # SLIP, GUESS = ESVESISlipAndGuess(fileterFromQT, filterFromQF, n_students, question_new_len)

    if flag_train:
        SLIP, GUESS = iRUpdateSandG(seed_LS[:n_students * len(q_matrix[0])], data, q_matrix, n_knowledge, n_students,
                                    n_questions)

        if not math.isnan(SLIP[0]) or not math.isnan(GUESS[0]):
            individual = updateIndividual_s_g(individual, SLIP, GUESS, n_students, n_knowledge, GENE_LENGTH,
                                              len_s_g)

    return individual, second_LS, second_LS_id


def local_search_test_ESVE_by_students(data, data_ESVE_id, seed_LS, q_matrix, n_students, n_questions,
                                       new_students):
    n_knowledge = len(q_matrix[0])
    Aarray = []
    # 从过滤出的
    filterFromQF = [0] * n_questions
    fileterFromQT = [0] * n_questions

    for i in range(new_students):
        # 第一步:获得Q矩阵
        QT, QF, QTid, QFid = acquireQTandQF(data[i], q_matrix, n_questions)

        QTreal, QFreal, fileterFromQT, filterFromQF = ESVECollisionDetection(QT, QF, n_knowledge, fileterFromQT,
                                                                             filterFromQF, QTid, QFid)

        # 第三步:从可靠的问题向量中估计学生向量
        # ESVEFromQrealByQuestion
        Ai = ESVEFromQrealByQuestion(QTreal, QFreal, n_knowledge)
        Aarray.append(Ai)

    Seed_LS_A = seed_LS[:n_students * len(q_matrix[0])]
    for i in range(len(data_ESVE_id)):
        for j in range(len(q_matrix[0])):
            if Aarray[i][j] != 999:
                Seed_LS_A[data_ESVE_id[i] * len(q_matrix[0]) + j] = Aarray[i][j]
    individual = updateIndividual_A_0(seed_LS, Seed_LS_A, n_students, n_knowledge)

    # TODO
    #   计算新个体的适应度
    # SLIP = np.array(slip)
    # GUESS = np.array(guess)
    #
    # label = []
    # for i in range(n_students):
    #     for j in range(n_questions):
    #         label.append(data[i][j])
    # bestA = acquireA(individual, n_students, n_knowledge)
    # bestYITA = acquireYITA(bestA, q_matrix, n_students, n_questions, n_knowledge)
    # bestX, Xscore = acquireX(n_students, n_questions, bestYITA, SLIP, GUESS)  #
    # predict = []
    # for i in range(n_students):
    #     for j in range(n_questions):
    #         predict.append(bestX[i][j])
    # predictScore = []
    # for i in range(n_students):
    #     for j in range(n_questions):
    #         predictScore.append(Xscore[i][j])
    # try:
    #     AUC = roc_auc_score(label, predict)
    # except ValueError:
    #     AUC = 0.5
    #
    # print("训练集auc = " + str(AUC))

    return individual  # 返回更新后的个体


def local_search_rectify_by_students(data, data_LS, data_LS_id, seed_LS, q_matrix, n_students, n_questions, n_knowledge,
                                     new_students, slip, guess, flag_train, GENE_LENGTH, len_s_g):
    A = acquireA(seed_LS, n_students, n_knowledge)
    YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    slip_ratio = slip
    guess_ratio = guess
    if flag_train:
        s, g = acquireSandG(n_students, seed_LS, n_knowledge, GENE_LENGTH, len_s_g)
        slip_ratio = s
        guess_ratio = g
    if not flag_train:
        slip_ratio = np.array(slip)
        guess_ratio = np.array(guess)
    X, X_score = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

    for i in range(new_students):
        # i_knowledge_clash_0 实际学生作对了此题， 但是预测做错了1
        i_knowledge_clash_0 = [0] * n_knowledge
        # i_knowledge_clash_1  实际学生做错了此题， 但是预测做对了
        i_knowledge_clash_1 = [0] * n_knowledge

        i_knowledge = seed_LS[data_LS_id[i] * n_knowledge: (data_LS_id[i] + 1) * n_knowledge]

        for j in range(n_questions):
            if data_LS[i][j] == 1 and X[data_LS_id[i]][j] == 0:
                for k in range(n_knowledge):
                    if q_matrix[j][k] == 1 and i_knowledge[k] == 0:
                        i_knowledge_clash_0[k] += 1
            elif data_LS[i][j] == 0 and X[data_LS_id[i]][j] == 1:
                for k in range(n_knowledge):
                    if q_matrix[j][k] == 1 and i_knowledge[k] == 1:
                        i_knowledge_clash_1[k] += 1

        # flag = 0 max_clash_0 翻转，  flag = 1 max_clash_1 翻转

        # if max_clash_0 != 0 and max_clash_0 >= max_clash_1:
        #     max_idx = i_knowledge_clash_0.index(max_clash_0)
        #     i_knowledge[max_idx] = 1
        #     # print("第" + str(i) + "个学生的第" + str(max_idx) + "个知识点变为1")
        # else:
        #     max_idx = i_knowledge_clash_1.index(max_clash_1)
        #     # print("第" + str(i) + "个学生的第" + str(max_idx) + "个知识点变为0")
        #     i_knowledge[max_idx] = 0
        #     flag = 1

        # 需要计算 翻转前与翻转后，哪个认知状态预测与真实作答相似状态多
        max_clash_0 = max(i_knowledge_clash_0)
        max_clash_1 = max(i_knowledge_clash_1)
        max_idx_0 = i_knowledge_clash_0.index(max_clash_0)
        max_idx_1 = i_knowledge_clash_1.index(max_clash_1)
        i_knowledge_seed = i_knowledge.copy()
        i_knowledge_0 = i_knowledge.copy()
        i_knowledge_1 = i_knowledge.copy()

        X_i_seed = acquireXForOneStudent(i_knowledge_seed, q_matrix, n_questions, n_knowledge, slip_ratio, guess_ratio)
        if max_clash_0 == 0:
            i_knowledge_0 = i_knowledge
            X_i_0 = X_i_seed
        else:
            i_knowledge_0[max_idx_0] = 1 - i_knowledge_0[max_idx_0]
            X_i_0 = acquireXForOneStudent(i_knowledge_0, q_matrix, n_questions, n_knowledge, slip_ratio, guess_ratio)

        if max_clash_1 == 0:
            i_knowledge_0 = i_knowledge
            X_i_1 = X_i_seed
        else:
            i_knowledge_1[max_idx_1] = 1 - i_knowledge_1[max_idx_1]
            X_i_1 = acquireXForOneStudent(i_knowledge_1, q_matrix, n_questions, n_knowledge, slip_ratio, guess_ratio)

        sum_seed = 0
        sum_0 = 0
        sum_1 = 0

        for j in range(n_questions):
            if data[i][j] == X_i_seed[j]:
                sum_seed += 1
            if data[i][j] == X_i_0[j]:
                sum_0 += 1
            if data[i][j] == X_i_1[j]:
                sum_1 += 1

        max_resemble = max(sum_seed, sum_0, sum_1)
        if max_resemble == sum_seed:
            pass
            # print("学生" + str(i) + "当前认知状态是最好的。")
        elif max_resemble == sum_0:
            # print("学生" + str(i) + " :0翻转")
            seed_LS[data_LS_id[i] * n_knowledge: (data_LS_id[i] + 1) * n_knowledge] = i_knowledge_0
        elif max_resemble == sum_1:
            # print("学生" + str(i) + " :1翻转")
            seed_LS[data_LS_id[i] * n_knowledge: (data_LS_id[i] + 1) * n_knowledge] = i_knowledge_1

        # seed_LS[data_LS_id[i] * n_knowledge: (data_LS_id[i] + 1) * n_knowledge] = i_knowledge

        if flag_train:
            SLIP, GUESS = iRUpdateSandG(seed_LS[:n_students * len(q_matrix[0])], data, q_matrix, n_knowledge,
                                        n_students,
                                        n_questions)
            if not math.isnan(SLIP[0]) or not math.isnan(GUESS[0]):
                seed_LS = updateIndividual_s_g(seed_LS, SLIP, GUESS, n_students, n_knowledge, GENE_LENGTH,
                                               len_s_g)

    return seed_LS


def updateIndividual_S_G(individual, data, q_matrix, n_knowledge, n_students, n_questions, GENE_LENGTH, len_s_g):
    A = individual[: n_students * n_knowledge]
    SLIP, GUESS = iRUpdateSandG(A, data, q_matrix, n_knowledge, n_students, n_questions)

    if not math.isnan(SLIP[0]) or not math.isnan(GUESS[0]):
        individual = updateIndividual_s_g(individual, SLIP, GUESS, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)

    return individual


def updateIndividual_global_guide(global_guide_ind, seed_LS, data_id, data, q_matrix, n_students,
                                  n_questions, n_knowledge, slip, guess, flag_train, GENE_LENGTH, len_s_g):
    A_global = acquireA(global_guide_ind, n_students, n_knowledge)
    YITA_global = acquireYITA(A_global, q_matrix, n_students, n_questions, n_knowledge)
    slip_global = slip
    guess_global = guess
    if flag_train:
        s_global, g_global = acquireSandG(n_students, global_guide_ind, n_knowledge, GENE_LENGTH, len_s_g)
        slip_global = s_global
        guess_global = g_global
    if not flag_train:
        slip_global = np.array(slip)
        guess_global = np.array(guess)
    X_global, X_global_score = acquireX(n_students, n_questions, YITA_global, slip_global, guess_global)

    A_seed = acquireA(seed_LS, n_students, n_knowledge)
    YITA_seed = acquireYITA(A_seed, q_matrix, n_students, n_questions, n_knowledge)
    slip_seed = slip
    guess_seed = guess
    if flag_train:
        s_seed, g_seed = acquireSandG(n_students, seed_LS, n_knowledge, GENE_LENGTH, len_s_g)
        slip_seed = s_seed
        guess_seed = g_seed
    if not flag_train:
        slip_seed = np.array(slip)
        guess_seed = np.array(guess)
    X_seed, X_seed_score = acquireX(n_students, n_questions, YITA_seed, slip_seed, guess_seed)

    for i in range(len(data_id)):
        sum_global = 0
        sum_seed = 0
        for j in range(n_questions):
            if X_global[data_id[i]][j] == data[data_id[i]][j]:
                sum_global += 1
            if X_seed[data_id[i]][j] == data[data_id[i]][j]:
                sum_seed += 1
        if sum_seed > sum_global:
            for k in range(n_knowledge):
                global_guide_ind[data_id[i] * n_knowledge + k] = seed_LS[data_id[i] * n_knowledge + k]
        else:
            for k in range(n_knowledge):
                seed_LS[data_id[i] * n_knowledge + k] = seed_LS[data_id[i] * n_knowledge + k]

    if flag_train:
        global_guide_ind = updateIndividual_S_G(global_guide_ind, data, q_matrix, n_knowledge, n_students, n_questions,
                                                GENE_LENGTH, len_s_g)

    return global_guide_ind


def add_population_by_global(ind, global_guide_ind, GENE_LENGTH):
    # mode == 0 此时取global个体中的position1到position2的部分，mode == 1 则取两边
    if random.random() < 0.5:
        mode = 0
    else:
        mode = 1
    position1 = random.randint(0, int(GENE_LENGTH / 3))
    position2 = random.randint(int(GENE_LENGTH / 3 * 2), GENE_LENGTH - 1)
    if mode == 0:
        ind[position1:position2] = global_guide_ind[position1: position2]
    else:
        ind[0:position1] = global_guide_ind[0:position1]
        ind[position2:GENE_LENGTH] = global_guide_ind[position2: GENE_LENGTH]
    return ind


def acquireXByInd(individual, q_matrix, n_students, n_questions, n_knowledge, slip, guess, flag_train, GENE_LENGTH,
                  len_s_g):
    A = acquireA(individual, n_students, n_knowledge)
    YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    Slip = slip
    Guess = guess
    if flag_train:
        s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
        Slip = s
        Guess = g
    if not flag_train:
        Slip = np.array(slip)
        Guess = np.array(guess)
    X, X_score = acquireX(n_students, n_questions, YITA, Slip, Guess)

    return X
