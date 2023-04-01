import copy
import math
import random
from queue import Queue
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.metrics import roc_auc_score

import train_model
from localSearch import *
from tools import *
from initializeFunction import *
import os


def IGA(tem_files, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess,
        data_patch_id, run_id, n_pop, flag_train, max_generations, len_s_g, alg_name='GA_NBC',
        data_name='Math_DMiC'):
    n_knowledge = n_knowledge_fine
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g
    else:
        GENE_LENGTH = n_students * n_knowledge

    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    pop = toolbox.Population(n=n_pop)

    def evaluate(individual):

        A = acquireA(individual, n_students, n_knowledge)
        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
        slip_ratio = slip
        guess_ratio = guess
        if flag_train:
            s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
            slip_ratio = s
            guess_ratio = g

        if flag_train == False:
            slip_ratio = np.array(slip)
            guess_ratio = np.array(guess)

        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

        label = []
        for s in range(n_students):
            for j in range(n_questions):
                label.append(data[s][j])
        predict = []
        for s in range(n_students):
            for j in range(n_questions):
                predict.append(X[s][j])
        try:
            AUC = roc_auc_score(label, predict)
        except ValueError:
            AUC = 0.5

        return (AUC),

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('cxUniform', tools.cxUniform, indpb=0.3)
    toolbox.register('cxTwoPoint', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)

    global_guide_ind = toolbox.Population(n=1)[0]
    global_guide_ind.fitness.values = evaluate(global_guide_ind)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    sample_knowledge = 6
    for ind in pop:
        if flag_train:
            ind = initial_population(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH, len_s_g,
                                     sample_knowledge)
        else:
            ind = initial_population_test(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH,
                                          len_s_g,
                                          sample_knowledge, slip, guess)
        ind.fitness.values = evaluate(ind)

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)

    tempTotalPop = []
    LS = 0
    NBC_cluster = []
    statistical_data = []
    coefficient_variation = []
    LSInvalid = 0

    for gen in range(1, max_generations + 1):

        sample_student = int((math.ceil(gen / (max_generations / 10)) / 10) * n_students)
        if (gen == 1):
            totalPop = pop
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)
        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)
            NBC_cluster.append(len(newg))
        except:
            pass
        Popth = 0
        for PopNodeList in newg:
            Popth += 1
            subPop = []
            for nodeId in PopNodeList.vs['label']:
                try:
                    subPop.append(totalPop[nodeId])
                except:
                    pass
            cxpb = 0.4329
            mutpb = 0.09351

            N_subPop = len(subPop)
            ind_seed = subPop[random.randint(0, N_subPop - 1)]
            seed_LS = toolbox.clone(ind_seed)
            seed_LS_tem = toolbox.clone(ind_seed)

            update_LS_id, data_all = acquireDataIsESVE(data, sample_student)
            LS += 1


            seed_LS, second_LS, second_LS_id = local_search_train_ESVE_by_students(data, data_all, update_LS_id,
                                                                                   seed_LS,
                                                                                   seed_LS_tem, q_matrix,
                                                                                   n_students,
                                                                                   n_questions, len(data_all),
                                                                                   flag_train, GENE_LENGTH,
                                                                                   len_s_g, slip, guess)
            seed_LS_tem = toolbox.clone(seed_LS)
            if len(second_LS_id) != 0:
                seed_LS = local_search_rectify_by_students(data, second_LS, second_LS_id, seed_LS, q_matrix,
                                                           n_students,
                                                           n_questions, n_knowledge, len(second_LS), slip,
                                                           guess,
                                                           flag_train, GENE_LENGTH, len_s_g)
                seed_LS.fitness.values = evaluate(seed_LS)
                seed_LS_tem.fitness.values = evaluate(seed_LS_tem)
                if float(seed_LS.fitness.values[0]) < float(seed_LS_tem.fitness.values[0]):
                    LSInvalid += 1
                    seed_LS = seed_LS_tem

            global_guide_ind_tem = toolbox.clone(global_guide_ind)
            global_guide_ind = updateIndividual_global_guide(global_guide_ind, seed_LS, update_LS_id, data,
                                                             q_matrix, n_students,
                                                             n_questions, n_knowledge, slip, guess,
                                                             flag_train, GENE_LENGTH, len_s_g)
            global_guide_ind.fitness.values = evaluate(global_guide_ind)

            if global_guide_ind.fitness.values < global_guide_ind_tem.fitness.values:
                    global_guide_ind = global_guide_ind_tem


            offspring = toolbox.select(subPop, N_subPop)
            offspring_Xor = []
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))

            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)

            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')
            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1
            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 10:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1

            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')

            tempTotalPop.extend(pop_selected)

        pop_selected_total = []
        num_selected_total = 1
        pop_selected_total.append(tempTotalPop[0])
        for i in range(1, len(tempTotalPop)):
            idx = pop_selected_total[num_selected_total - 1]
            dis = sum([ch1 != ch2 for ch1, ch2 in zip(tempTotalPop[i], idx)])
            if dis >= 10:
                pop_selected_total.append(tempTotalPop[i])
                num_selected_total = num_selected_total + 1

        tempTotalPop = tools.selBest(pop_selected_total, n_pop, fit_attr='fitness')
        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(tempTotalPop), **record)
        if len(tempTotalPop) < n_pop:
            pop_add = toolbox.Population(n=(n_pop - len(tempTotalPop)))
            sample_knowledge_add = int((n_knowledge / 3) * 2)
            for ind in pop_add:
                # ind = add_population_by_global(ind, global_guide_ind, GENE_LENGTH)
                if flag_train:
                    ind = add_population(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH,
                                         len_s_g, sample_knowledge_add)
                else:
                    ind = add_population_test(ind, data, q_matrix, n_students, n_questions, n_knowledge, GENE_LENGTH,
                                              len_s_g, sample_knowledge_add, slip, guess)
                ind.fitness.values = evaluate(ind)
            tempTotalPop.extend(pop_add)
            print("第" + str(gen) + "代增加的个体数为" + str(len(pop_add)))
        statistical_data.append((logbook.select("max")[gen] - logbook.select("min")[gen]) / logbook.select("avg")[gen])
        coefficient_variation.append(logbook.select("std")[gen] / logbook.select("avg")[gen])

    resultPop = tempTotalPop

    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)
            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)
            return A

    index = np.argmax([ind.fitness for ind in resultPop])

    if flag_train:
        slip, guess = decode(global_guide_ind)

    gen = logbook.select('gen')
    fit_maxs = logbook.select('max')

    # print("fit_max: ",fit_maxs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    # ax.plot(gen[1:], fit_avgs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    ax.legend(loc='best')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

    fig.tight_layout()

    if not flag_train:
        path_pre = tem_files + '/test/'
    else:
        path_pre = tem_files + '/train/'
    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' 共' + str(max_generations) + '代：   第' + str(
        data_patch_id) + '折' + '第' + str(run_id) + '次训练' + '.png')

    print(str(max_generations) + "代局部搜索次数为：" + str(LS))
    print(str(max_generations) + "代局部搜索无效次数为：" + str(LSInvalid))
    print("global_guide_ind的AUC为：" + str(global_guide_ind.fitness.values[0]))
    print("种群中最优个体的AUC为：" + str(resultPop[index].fitness.values[0]))
    if resultPop[index].fitness.values[0] > global_guide_ind.fitness.values[0]:
        print("种群中产生最优个体！")
        global_guide_ind = resultPop[index]
    else:
        print("global_guide_ind为最优个体！")
    return resultPop, logbook, slip, guess, global_guide_ind


