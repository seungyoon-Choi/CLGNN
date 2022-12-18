import time 
import random 
import argparse
import random

import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 
from model import GCN, GAT

from replay import *
from utils import *


def main(args):
    # meta settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')

    # load the data
    adj_ori, features_ori, labels_ori, train, total_classes, partition, train_per_class, test_per_class, class_idx = load_data(args, dataset = args.dataset, classes_per_task=args.classes_per_task)
    features_ori = features_ori.to(device)
    labels_ori = labels_ori.to(device)

    #Train
    replay_buffer = None
    handle = 0

    handled = []
    accs = []
    acc = {}
    current_accs = []
    buffer = {}

    for task in range(len(train)):

        for cla in partition[task]:
            handle += 1
            for node in class_idx[cla]:
                handled.append(node)
            buffer[cla] = []
        
        acc[task] = []

        matching_index = {} #changed -> original
        matching_index_inv = {} #original -> changed
        for idx in range(len(handled)):
            matching_index[idx] = handled[idx]
            matching_index_inv[int(handled[idx])] = torch.tensor(idx)


        #새로운 index에서 train_index와 test_index set
        train_idx = []
        val_idx = []
        test_idx = []
        replay_idx = []
        current_idx = []
        for i in partition[task]:
            train_idx += [matching_index_inv[int(x)] for x in train_per_class[i]]
            current_idx += [matching_index_inv[int(x)] for x in train_per_class[i]]
        if replay_buffer != None:
            train_idx += [matching_index_inv[int(x)] for x in replay_buffer]
            replay_idx += [matching_index_inv[int(x)] for x in replay_buffer]

        total_data = 0

        for tasks in range(task+1):
            test_idx.append([])
            val_idx.append([])
            for i in partition[tasks]:
                test_idx[tasks] += [matching_index_inv[int(x)] for x in test_per_class[i]]
                total_data += len(train_per_class[i])

        #update adjacency
        adj = adj_masking(adj_ori, handled, device)
        features = feature_masking(features_ori, handled)
        labels = label_masking(labels_ori, handled)

        if args.structure == 'yes':
            # Link Prediction
            if task == 0:
                model_lp, adj = train_linkpred(features, adj, device)
            if task != 0:
                model_lp, adj = train_linkpred(features, adj, device)

        if task == len(partition)-1:
            stat_degree = []
            stat_homophily = []
            for buffer in replay_idx:
                deg, hom = degree_homophily(buffer, adj, labels)
                stat_degree.append(deg)
                stat_homophily.append(hom)
            deg_mean = np.mean(stat_degree)
            deg_std = np.std(stat_degree)
            hom_mean = np.mean(stat_homophily)
            hom_std = np.std(stat_homophily)


        if task == 0:
            # parameter intialization
            N = features.size(0) # num_of_nodes
            F = features.size(1) # num_of_features
            H = args.hidden # hidden nodes
            C = args.classes_per_task

            if args.model == 'GCN':
                network = GCN(F, H, C, args.dropout).to(device)                
                
            else:
                network = GAT(F, H, C, N, args.dropout, args.alpha, args.n_heads).to(device)
            
            optimizer = optim.Adam(network.parameters(), lr = args.lr, weight_decay = args.weight_decay)
            criterion = nn.CrossEntropyLoss()

        if task != 0:
            if args.model == 'GCN':
                network.load_state_dict(torch.load('checkpoints/%d.pt' %(task-1)))
                #Parameter Expand
                param_expand = torch.rand(H,C).to(device)
                new_param = torch.cat((network.layer2.W, param_expand),1)
                network.layer2.W = nn.Parameter(new_param)
            elif args.model == 'GAT':
                network.load_state_dict(torch.load('checkpoints/%d.pt' %(task-1)))
                # previous task parameters
                weight_previous = network.layer2.attentions[0].W.weight
                att_previous = network.layer2.attentions[0].a_T.weight
                # randomly initialized parameter expand
                weight_expand = torch.rand(C, H*args.n_heads).to(device)
                att_expand = torch.rand(1, 2*C).to(device)
                # Combine together
                weight_new = torch.cat((weight_previous, weight_expand), 0)
                att_new = torch.cat((att_previous, att_expand), 1)
                # Last layer expansion (parameters are randomly initialized)
                network.layer2.attentions[0].W = nn.Linear(H*args.n_heads, C*(task+1), bias=False)
                network.layer2.attentions[0].a_T = nn.Linear(2*C*(task+1), 1, bias=False)
                # Replace the parameter
                network.layer2.attentions[0].W.weight = nn.Parameter(weight_new)
                network.layer2.attentions[0].a_T.weight = nn.Parameter(att_new)

        for epoch in range(args.epochs):
            t = time.time()
            network.train()

            preds = network(features, adj)

            if task == 0:
                train_loss = criterion(preds[train_idx], labels[train_idx])  #train[task] index modify
            elif task != 0:
                current_loss = criterion(preds[current_idx], labels[current_idx])
                replay_loss = criterion(torch.index_select(preds.to(device), 0, torch.tensor(replay_idx).to(device)), torch.index_select(labels.to(device), 0, torch.tensor(replay_idx).to(device)))
                
                if args.structure == 'no':
                    beta = len(replay_idx)/(len(current_idx)+len(replay_idx))
                else:
                    beta = 0.25

                if len(replay_idx) == 0:
                    train_loss = current_loss
                else:
                    train_loss = beta * current_loss + (1-beta) * replay_loss

            train_acc = accuracy(preds[train_idx], labels[train_idx])
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print('[%d/%d] train loss : %.4f | train acc %.2f%% | time %.3fs'
                %(epoch+1, args.epochs, train_loss.item(), train_acc.item() * 100, time.time() - t))

        torch.save(network.state_dict(), 'checkpoints/%d.pt' %(task))

        # update replay buffer
        if task == 0:
            mean_features = count = dist = cm = replay = distances = distances_mean = homophily = degree = None
        replay_buffer, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree = update_replay(args.replay, network, matching_index, matching_index_inv, args.memory_size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, args.alpha_dist, args.beta_homo, args.gamma_deg, args.one_per_class, train_per_class, args.clustering, args.k, args.distance, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree)
        
        
        # Test
        with torch.no_grad():
            network.eval()
            preds = network(features, adj)
            for tasks in range(task+1):
                if args.incremental == 'class':
                    test_acc = accuracy(preds[test_idx[tasks]], labels[test_idx[tasks]])
                    acc[tasks].append(test_acc)
                    if task == len(train)-1:
                        current_accs.append(test_acc)
                elif args.incremental == 'task':
                    test_acc = task_accuracy(tasks, preds[test_idx[tasks]], labels[test_idx[tasks]], class_per_task = args.classes_per_task)
                    acc[tasks].append(test_acc)
                    if task == len(train)-1:
                        current_accs.append(test_acc)
                print('%d. Test Accuracy for task %s : %.2f'%(task+1, str(tasks+1), test_acc * 100))
            accs.append(test_acc)
    
    print('Average Performance : %.2f'%(average_performance(acc)))
    print('Average Performance for last task : %.2f'%(current_performance(acc)))
    print('Forgetting Performance : %.2f'%(forget_performance(acc)))
    

def average_performance(acc):
    sum = 0
    for i in range(len(acc)):
        sum += acc[i][0]
    return round((sum / len(acc)).item() * 100, 2)

def current_performance(acc):
    sum = 0
    for i in range(len(acc)):
        sum += acc[i][-1]
    return round((sum / len(acc)).item() * 100, 2)

def forget_performance(acc):
    sum = 0
    count = 0
    for task in acc.keys():
        sum += (acc[task][0]-acc[task][-1])
        count += (len(acc[task])-1)
    return round((sum / count).item() * 100, 2)
    


if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_per_task', type=int, default=2, help='classes per task')
    parser.add_argument('--replay', type=str, default='MFe', choices=['random', 'MFf', 'MFe', 'CMf', 'CMe', 'C_Me', 'C_Mf', 'MF_hd'], help='replay method')
    parser.add_argument('--memory_size', type=int, default = 40, help='replay buffer size')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='cora', choices=['corafull','cora','citeseer'], help='Dataset to train.')
    parser.add_argument('--model', type=str, default='GAT', choices=['GCN','GAT'], help='Model to train.')
    parser.add_argument('--clustering', type=str, default='yes', choices=['yes', 'no'], help='diversity')
    parser.add_argument('--k', type=int, default=4, help = 'num of clusters of each class')
    parser.add_argument('--seed_shuffle', type=int, default=4, help = 'task sequence shuffle')
    parser.add_argument('--incremental', type=str, default='class', choices=['class', 'task'], help='incremental type')
    parser.add_argument('--distance', type=float, default=0.2, help='distance threshold in CM')
    parser.add_argument('--one_per_class', type=bool, default=False, help='Whether to replay one per class')
    parser.add_argument('--alpha_dist', type=float, default = 1, help='ratio of MF')
    parser.add_argument('--beta_homo', type=float, default=1, help='ratio of homophily')
    parser.add_argument('--gamma_deg', type=float, default=1, help='ratio of degree')
    parser.add_argument('--structure', type=str, default='no', choices=['yes', 'no'], help='structure learning?')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)