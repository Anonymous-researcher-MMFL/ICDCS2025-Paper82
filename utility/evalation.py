import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from collections import defaultdict
import random
def evaluation(model, data, batch_size, device, args):
    model.eval()
    # split data into validation set (first 30%) and test set (last 70%)
    # Calculate the number of validation samples
    num_val_samples = int(len(data) * 0.3)
    indices = list(range(len(data)))

    # Shuffle indices if you want to randomize the validation set
    # random.shuffle(indices)

    # Select the indices for the validation and test sets
    val_indices = indices[:num_val_samples]
    test_indices = indices[num_val_samples:]

    if args is None:
        # do nothing, use all data
        pass
    else:
        if args.validation is True:
            data = Subset(data, val_indices)
        else:
            data = Subset(data, test_indices)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    running_loss = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        accuracy = correct/total
        loss = running_loss/total

    return accuracy, loss


def get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info, tasks_data_idx, global_models, device, batch_size, venn_matrix, freshness, localLoss, fresh_ratio):

    if freshness is False:
        localLoss = np.zeros((task_number, num_clients))
        for cl in range(num_clients):
            for task in range(task_number):
                if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                        cl])  # or iid_partition depending on your choice
                elif type_iid[task] == 'noniid':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                        cl])  # or iid_partition depending on your choice
                accu, loss = evaluation(model=global_models[task], data=client_data,
                                        batch_size=batch_size, device=device, args=None)  # use all data
                # accu = localAccResults[task, cl, round-1]
                # localLossResults[task, cl, round] = loss
                # localAcc[task, cl] = accu
                localLoss[task, cl] = loss * venn_matrix[task, cl] # if venn_matrix[task, cl] == 0, then the loss is also 0
    elif freshness is True:
        subset_ratio = fresh_ratio
        fresh_factor = 1 # np.e**(-0.1)
        # sample some task to update loss
        for cl in range(num_clients):
            for task in range(task_number):
                # if random value is larger than subset_ratio, then skip this task
                if venn_matrix[task, cl] == 0 or random.random() > subset_ratio:
                    localLoss[task, cl] = localLoss[task, cl] / fresh_factor * venn_matrix[task, cl]
                    continue
                # else update the loss and some matrix
                if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                        cl])  # or iid_partition depending on your choice
                elif type_iid[task] == 'noniid':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                        cl])  # or iid_partition depending on your choice
                accu, loss = evaluation(model=global_models[task], data=client_data,
                                        batch_size=batch_size, device=device, args=None)  # use all data
                # update the loss
                localLoss[task, cl] = loss
    return localLoss

def get_local_acc(task_number, num_clients, task_type, type_iid, tasks_data_info, tasks_data_idx, global_models, device, batch_size, venn_matrix, freshness, localLoss, fresh_ratio):
    if freshness is False:
        localLoss = np.zeros((task_number, num_clients))
        for cl in range(num_clients):
            for task in range(task_number):
                if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                        cl])  # or iid_partition depending on your choice
                elif type_iid[task] == 'noniid':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                        cl])  # or iid_partition depending on your choice
                accu, loss = evaluation(model=global_models[task], data=client_data,
                                        batch_size=batch_size, device=device, args=None)  # use all data
                # accu = localAccResults[task, cl, round-1]
                # localLossResults[task, cl, round] = loss
                # localAcc[task, cl] = accu
                localLoss[task, cl] = loss * venn_matrix[task, cl] # if venn_matrix[task, cl] == 0, then the loss is also 0
    elif freshness is True:
        subset_ratio = fresh_ratio
        fresh_factor = 1 # np.e**(-0.1)
        # sample some task to update loss
        for cl in range(num_clients):
            for task in range(task_number):
                # if random value is larger than subset_ratio, then skip this task
                if venn_matrix[task, cl] == 0 or random.random() > subset_ratio:
                    localLoss[task, cl] = localLoss[task, cl] / fresh_factor * venn_matrix[task, cl]
                    continue
                # else update the loss and some matrix
                if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                        cl])  # or iid_partition depending on your choice
                elif type_iid[task] == 'noniid':
                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                        cl])  # or iid_partition depending on your choice
                accu, loss = evaluation(model=global_models[task], data=client_data,
                                        batch_size=batch_size, device=device, args=None)  # use all data
                # update the loss
                localLoss[task, cl] = 1-accu
    return localLoss


def group_fairness_evaluation(model, data, batch_size, device, args):
    model.eval()
    # split data into validation set (first 30%) and test set (last 70%)
    # Calculate the number of validation samples
    num_val_samples = int(len(data) * 0.3)
    indices = list(range(len(data)))

    # Shuffle indices if you want to randomize the validation set
    # random.shuffle(indices)

    # Select the indices for the validation and test sets
    val_indices = indices[:num_val_samples]
    test_indices = indices[num_val_samples:]

    if args is None:
        # do nothing, use all data
        pass
    else:
        if args.validation is True:
            data = Subset(data, val_indices)
        else:
            data = Subset(data, test_indices)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    running_loss = 0
    total = 0

    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    label_positive = defaultdict(int)
    label_true_positive = defaultdict(int)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Record accuracy for each label and true positive rates
            for label in torch.unique(labels):
                label_mask = (labels == label)
                label_correct[label.item()] += (predicted[label_mask] == labels[label_mask]).sum().item()
                label_total[label.item()] += label_mask.sum().item()

                # True positives for the label
                label_positive[label.item()] += label_mask.sum().item()
                label_true_positive[label.item()] += (predicted[label_mask] == labels[label_mask]).sum().item()

    accuracy = correct / total
    loss = running_loss / total

    # Calculate accuracy for each label
    label_accuracies = {label: label_correct[label] / label_total[label] for label in label_total}

    # Calculate true positive rates for each label
    label_tprs = {label: label_true_positive[label] / label_positive[label] for label in label_positive}

    # Calculate Equal Opportunity Difference
    eod = max(label_tprs.values()) - min(label_tprs.values())

    # Calculate the difference in accuracy as a group fairness measure
    accuracy_diff = max(label_accuracies.values()) - min(label_accuracies.values())

    # Save the results using pickle
    results = {
        'accuracy': accuracy,
        'loss': loss,
        'label_accuracies': label_accuracies,
        'accuracy_diff': accuracy_diff,
        'label_tprs': label_tprs,
        'eod': eod
    }

    return results