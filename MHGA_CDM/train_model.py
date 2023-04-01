from myAlgorithms import *


def trainModel(tem_files, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix,
               data_patch_id, run_id=1, n_pop=50,
               max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):
    flag_train = True
    slip = []
    guess = []

    resultPop, logbook, slip, guess, global_guide_ind = IGA(tem_files, n_students, n_questions, n_knowledge_coarse,
                                                            n_knowledge_fine, data,
                                                            q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                            flag_train, max_generations, len_s_g, alg_name, data_name)

    # now = time.time()
    # local_time = time.localtime(now)
    # date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    # print(date_format_localtime)

    return slip, guess
