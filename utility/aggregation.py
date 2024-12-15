import torch
import numpy as np
import utility.optimal_sampling as optimal_sampling
import copy
import pickle

def federated(models_state_dict, local_data_nums, aggregation_mtd, numUsersSel):

    global_state_dict = models_state_dict[0].copy()
    global_keys = list(global_state_dict.keys())

    for key in global_keys:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # Sum the state_dicts of all client models
    for i, model_state_dict in enumerate(models_state_dict):
        for key in global_keys:
            if aggregation_mtd=='pkOverSumPk':
                global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]
                # W_global = sum_
            elif aggregation_mtd=='numUsersInv':
                global_state_dict[key] += 1/len(local_data_nums)  * model_state_dict[key]
                # the above means 1/ num users
            #global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]

    return global_state_dict

def federated_prob(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, tasks_local_training_loss, lr):

    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha
    N = args.num_clients
    L = 1
    dis_s = local_data_num
    denominator = 0
    # aggregate
    if (args.fairness == 'notfair'):
        denominator = 1

        for i, gradient_dict in enumerate(models_gradient_dict):
            d_i = dis_s[chosen_clients[i]]
            for key in global_keys:
                global_weights_dict[key] -= (d_i / p_list[i]) * gradient_dict[key] / denominator
    elif args.fairness == 'taskfair':
        # get f_s and max gradient (H_s)
        f_s = 0
        H_s = 0
        assert len(tasks_local_training_loss) == N
        for i in range(N):
            d_is = dis_s[i]
            f_s += tasks_local_training_loss[i] * d_is
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = dis_s[chosen_clients[i]]
            norm = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) ** 0.5 * L
            if H_s < norm*d_is:
                H_s = norm*d_is
        denominator = (alpha-1) * (N * H_s)**2 + f_s * L
        #print(denominator)
        #print(f_s)
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = dis_s[chosen_clients[i]]
            for key in global_keys:
                global_weights_dict[key] -= d_is / p_list[i] * f_s * gradient_dict[key]*L / denominator
    else:
        print("aggregation wrong!")
        exit(1)

    return global_weights_dict


def federated_stale(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, old_global_weights, decay_beta, allocation_result, task_index, save_path):
    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha
    N = args.num_clients
    L = 1
    dis_s = local_data_num
    total_rounds = len(allocation_result)

    if args.MILA is True:  # no problem, because MIFA doesn't have optimal sampling, so we get original h_i without beta here
        clients_num = len(dis_s)
        for i in range(clients_num):
            if i not in chosen_clients:
                for key in global_keys:
                    global_weights_dict[key] -= dis_s[i] * old_global_weights[i][key]
            else:
                for key in global_keys:
                    global_weights_dict[key] -= dis_s[i] * models_gradient_dict[chosen_clients.index(i)][key]
    else:
        # other method, FedVARP, FedStale, our methods
        # For FedVARP(args.optimal_sampling is False),
        # decay_beta_record is always totally 0, old_global_weights is the original
        # FedStale has args.skipOS as True.
        # dict, key is client index, value is delta_
        for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
            d_i = dis_s[chosen_clients[i]]
            h_i = old_global_weights[chosen_clients[i]]
            for key in global_keys:
                # if we use summation window, then should minus h_i * window_size
                global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])
        clients_num = len(dis_s)
        for i in range(clients_num):
            d_i = dis_s[i]
            h_i = old_global_weights[i]
            for key in global_keys:
                global_weights_dict[key] -= d_i * h_i[key]
    return global_weights_dict