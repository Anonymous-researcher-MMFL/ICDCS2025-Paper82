import numpy as np
def initialize_group(client_num, group_num, label_info):
    # input
    # client_num: number of clients
    # group_num: number of groups
    # label_info: list of labels of clients. label_info[i] is the available labels of client i
    # output
    # group_clients: list of clients in each group
    # group_clients[i] is a list of clients in group i

    # fin max class idx
    group_labels = [[0,1,2,3,4],
                    [5,6,7,8,9]]
    # group number should be 2
    g1_list = []
    g2_list = []
    for clent_idx in range(client_num):
        client_labels = label_info[clent_idx]
        g1 = sum([1 for label in client_labels if label in group_labels[0]])
        g2 = sum([1 for label in client_labels if label in group_labels[1]])
        if g1 > g2: # assign to group 1
            g1_list.append(clent_idx)
        else:
            g2_list.append(clent_idx)
    group_clients = [g1_list, g2_list]
    return group_clients


def index_sampling(available_clients, sample_num):
    # input
    # available_clients: list of clients that can be sampled
    # sample_num: number of clients to be sampled
    # output
    # chosen_clients: list of clients that are sampled
    # chosen_clients[i] is the index of the i-th sampled client
    chosen_clients = np.random.choice(available_clients, sample_num, replace=False)
    return chosen_clients

def alpha_fair_new_task(alpha, loss_list):
    # input
    # alpha: alpha value in alpha-fairness
    # loss_list: list of loss values of tasks. loss[i] is the loss of task i
    # output
    # task_idx: index of the task that is selected
    # note: this task_idx is not the real idx of the task, but the idx of the task in the loss_list
    loss_list = np.array(loss_list)
    loss_list = loss_list ** alpha
    prob = loss_list / np.sum(loss_list)
    task_idx = np.random.choice(np.arange(len(loss_list)), p=prob)
    return task_idx


def gradient_similarity(total_gradient_each_task):
    # input:
    # total_gradient_each_task: list of gradients of clients for a specific task.
    # total_gradient_each_task[i] is the gradient of the i-th client
    # output:
    # similarity_martrix: matrix of similarity between clients
    client_num = len(total_gradient_each_task)
    gradient_dict = total_gradient_each_task.state_dict()
    global_keys = list(gradient_dict.keys())
    similarity_martrix = np.ones((client_num, client_num))
    for i in range(client_num):
        for j in range(i+1, client_num):
            diff = gradient_dict[i] - gradient_dict[j]
            similarity_martrix[i][j] = np.linalg.norm(diff)
            similarity_martrix[j][i] = similarity_martrix[i][j]
    return similarity_martrix


def clustering_similarity(similarity_matrices):
    # input:
    # similarity_matrices: list of similarity matrices of tasks
    # similarity_matrices[i] is the similarity matrix of task i
    # output:
    # cluster_list: list of clusters
    # cluster_list[i] includes all clients belong to the i-th cluster

    # aggregate similarity matrix
    total_similarity_matrix = np.zeros_like(similarity_matrices[0])
    for matrix in similarity_matrices:
        total_similarity_matrix += matrix
    # clustering
    from sklearn.cluster import SpectralClustering
    cluster_num = 3
    # convert similarity to distance
    # total_similarity_matrix = np.max(total_similarity_matrix) - total_similarity_matrix
    # spectral clustering doesn't need to convert similarity to distance
    # kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(total_similarity_matrix)
    clustering = SpectralClustering(n_clusters=cluster_num,
                                    affinity='precomputed',
                                    assign_labels='kmeans',
                                    random_state=0).fit(total_similarity_matrix)
    cluster_labels = clustering.labels_

    # Prepare a list to hold items for each cluster
    cluster_list = [[] for _ in range(max(cluster_labels) + 1)]

    # Assign items to their respective clusters
    for item_index, cluster_label in enumerate(cluster_labels):
        cluster_list[cluster_label].append(item_index)
    return cluster_list