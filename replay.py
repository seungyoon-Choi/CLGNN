import random
import matplotlib.pyplot as plt
import random

import numpy as np 
import torch
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans



def update_random(train_per_class, matching_index_inv, partition, task, size, replay_buffer, total_data, one_per_class):
    purified = {}
    for tasks in range(task+1):
        for i in partition[tasks]:
            purified[i] = [matching_index_inv[int(x)] for x in train_per_class[i]]
            if one_per_class:
                proportion = 1
            else:
                proportion = min(int(len(purified[i]) / total_data * size), len(purified[i]))
            memo = random.sample(range(len(purified[i])), k = proportion)
            memory = [purified[i][idx] for idx in memo]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    return replay_buffer


def update_MF_hd(network, type, matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, alpha, homophily, beta, degree, gamma, one_per_class, train_per_class, clustering, k):

    if type == 'embedding':
        embeds = network.get_embeds(features, adj)[:]

    purified = [matching_index_inv[int(x)] for x in train[task]]

    if clustering == 'yes':
        purified_per_class = {}

        for cla in partition[task]:
            purified_per_class[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]
            mean_features[cla] = []
            count[cla] = []
            dist[cla] = []
            homophily[cla] = []
            degree[cla] = []

            for i in range(k):
                #mean_features[cla] += [torch.zeros(embeds.size(1)).to(device)]
                mean_features[cla] += [torch.zeros(len(features[0])).to(device)]
                count[cla].append([])
                dist[cla].append([])
                homophily[cla].append([])
                degree[cla].append([])

            if type == 'feature':
                #cluster_ids_x, cluster_centers = kmeans(X=features[purified_per_class[cla]], num_clusters=k, distance='euclidean', device=device)
                cluster = KMeans(n_clusters = k, random_state=1, n_init="auto").fit(features[purified_per_class[cla]])
                cluster_ids_x = cluster.labels_
                for j in range(len(cluster_ids_x)):
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])
            elif type == 'embedding':
                #cluster_ids_x, cluster_centers = kmeans(X=embeds[purified_per_class[cla]], num_clusters = k, distance = 'euclidean', device = device)
                cluster = KMeans(n_clusters = k, random_state=1, n_init="auto").fit(embeds[purified_per_class[cla]])
                cluster_ids_x = cluster.labels_
                for j in range(len(cluster_ids_x)):
                    #mean_features[cla][cluster_ids_x[j]] += embeds[purified_per_class[cla][j]]
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                embeds = embeds
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])

        for i in partition[task]:
            #for j in range(k):
            for j in range(len(count[i])):
                for k in range(len(count[i][j])):
                    if type == 'feature':
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])
                    elif type == 'embedding':
                        #dist[i][j].append([count[i][j][k], float(sum(pow(embeds[count[i][j][k]]-mean_features[i][j],2)))])
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])
                    
                    deg, hom = degree_homophily(count[i][j][k], adj,labels)
                    #homophily update
                    homophily[i][j].append([count[i][j][k], hom])
                    #degree update
                    degree[i][j].append([count[i][j][k], deg])


        for i in mean_features.keys():
            #for j in range(k):
            for j in range(len(count[i])):
                distt = minmax(np.array([dist[i][j][ind][1] for ind in range(len(dist[i][j]))]))
                homomo = minmax(np.array([homophily[i][j][ind][1] for ind in range(len(homophily[i][j]))]))
                degreee = minmax(np.array([degree[i][j][ind][1] for ind in range(len(degree[i][j]))]))

                total = alpha * distt - beta * homomo - gamma * degreee

                if one_per_class:
                    proportion=1
                else:
                    proportion = min(int(len(count[i][j]) / total_data * size), len(count[i][j]))
                ind = np.argpartition(total, proportion)[:proportion]
                memory = [dist[i][j][idx][0] for idx in ind]
                memory = torch.from_numpy(np.array(memory))
                if replay_buffer == None:
                    replay_buffer = memory
                else:
                    replay_buffer = torch.cat((replay_buffer, memory),0)

    elif clustering == 'no':
        #각 class당 feature의 평균
        for cla in partition[task]:
            if type == 'feature':
                mean_features[cla] = torch.zeros(len(features[0])).to(device)
            elif type == 'embedding':
                mean_features[cla] = torch.zeros(embeds.size(1)).to(device)
            count[cla] = []
            dist[cla] = []

        for i in range(len(purified)):
            if type == 'feature':
                mean_features[int(labels[purified[i]])] += features[purified[i]]
            elif type == 'embedding':
                mean_features[int(labels[purified[i]])] += embeds[purified[i]]
            count[int(labels[purified[i]])].append(purified[i])

        for i in partition[task]:
            mean_features[i] /= len(count[i]) #평균값으로 환산
            for j in range(len(count[i])):
                if type == 'feature':
                    dist[i].append([count[i][j], float(sum(pow(features[count[i][j]]-mean_features[i],2)))])
                elif type == 'embedding':
                    dist[i].append([count[i][j], float(sum(pow(embeds[count[i][j]]-mean_features[i],2)))]) #dist 계산
            
        for i in mean_features.keys():
            distt = np.array([dist[i][ind][1] for ind in range(len(dist[i]))])
            if one_per_class:
                proportion = 1
            else:
                proportion = min(int(len(count[i]) / total_data * size), len(count[i]))
            ind = np.argpartition(distt, proportion)[:proportion]
            memory = [dist[i][idx][0] for idx in ind]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, mean_features, count, dist, homophily, degree

def coverage_max(network, type, train_per_class, matching_index_inv, size, replay, total_data, partition, task, features, adj, cm, distance, one_per_class, distances, distances_mean):
    
    if type == 'embedding':
        embeds = network.get_embeds(features, adj)[:]
    
    purified = {}
    replay_buffer = None

    for cla in partition[task]:
        cm[cla] = []
        purified[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]
        distances[cla] = []
        distances_mean[cla] = []

    for cla in partition[task]:
        if one_per_class:
            proportion = 1
        else:
            proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))

        count = 0
        memory = []
        cover = []


        for idx in range(len(purified[cla])):
            if type == 'feature':
                dist = pow(features[purified[cla][idx]]-features[purified[cla][:idx]+purified[cla][idx+1:]],2)
            elif type == 'embedding':
                dist = pow(embeds[purified[cla][idx]]-embeds[purified[cla][:idx]+purified[cla][idx+1:]],2)
            counts = np.sqrt(np.sum(dist.cpu().detach().numpy(), 1))
            mean = np.mean(counts)
            distances[cla].extend(counts[idx:])
            distances_mean[cla].append(mean)

        while count < proportion:
            cm[cla] = []
            
            for idx in range(len(purified[cla])):
                if purified[cla][idx] in cover:
                    #pass
                    cm[cla].append([train_per_class[cla][idx], -1, []])
                else:
                    if type == 'feature':
                        dist = pow(features[purified[cla][idx]]-features[purified[cla]],2)
                    elif type == 'embedding':
                        dist = pow(embeds[purified[cla][idx]]-embeds[purified[cla]],2)
                    
                    counts = np.sqrt(np.sum(dist.cpu().detach().numpy(),1))

                    cm[cla].append([train_per_class[cla][idx], len(list(set(np.where(counts<distance)[0]) - set(cover))), list(set(np.where(counts<distance)[0]) - set(cover))])
            centrality = np.array([cm[cla][ind][1] for ind in range(len(cm[cla]))])

            ind = centrality.argmax()
            memory.append(train_per_class[cla][ind])
            if len(memory) > 1 and memory[-1] in memory[:-1]:
                del memory[-1]
                cover = [matching_index_inv[int(x)] for x in memory]
            else:
                cover = list(set(cover) | set(cm[cla][ind][2]))
                count += 1

        memory = torch.from_numpy(np.array(memory))

        replay[cla] = memory

    for i in range(task+1):
        for cla in partition[i]:
            if one_per_class:
                proportion = 1
            else:
                proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))
            
            if replay_buffer == None:
                replay_buffer = replay[cla][:proportion]
            else:
                replay_buffer = torch.cat((replay_buffer, replay[cla][:proportion]), 0)

    return replay_buffer, replay, cm, distances, distances_mean


def update_MF(network, type, matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, one_per_class, train_per_class, clustering, k):

    if type == 'embedding':
        embeds = network.get_embeds(features, adj)[:]

    purified = [matching_index_inv[int(x)] for x in train[task]]

    if clustering == 'yes':
        purified_per_class = {}

        for cla in partition[task]:
            purified_per_class[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]
            mean_features[cla] = []
            count[cla] = []
            dist[cla] = []

            if type == 'feature':
                cluster_ids_x, cluster_centers = kmeans(X=features[purified_per_class[cla]], num_clusters=k, distance='euclidean', device=device)
                #cluster = KMeans(n_clusters = k, random_state=1, n_init=10).fit(features[purified_per_class[cla]].cpu())
                #cluster_ids_x = cluster.labels_
                for i in range(k):
                    mean_features[cla] += [torch.zeros(len(features[0])).to(device)]
                    count[cla].append([])
                    dist[cla].append([])
                for j in range(len(cluster_ids_x)):
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])
            elif type == 'embedding':
                cluster_ids_x, cluster_centers = kmeans(X=embeds[purified_per_class[cla]], num_clusters = k, distance = 'euclidean', device = device)
                #cluster = KMeans(n_clusters = k, random_state=1, n_init=10).fit(embeds[purified_per_class[cla]].cpu())
                #cluster_ids_x = cluster.labels_
                for i in range(k):
                    #mean_features[cla] += [torch.zeros(embeds.size(1)).to(device)]
                    mean_features[cla] += [torch.zeros(len(features[0])).to(device)]
                    count[cla].append([])
                    dist[cla].append([])
                for j in range(len(cluster_ids_x)):
                    #mean_features[cla][cluster_ids_x[j]] += embeds[purified_per_class[cla][j]]
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                embeds = embeds
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])

        for i in partition[task]:
            #for j in range(k):
            for j in range(len(count[i])):
                for k in range(len(count[i][j])):
                    if type == 'feature':
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])
                    elif type == 'embedding':
                        #dist[i][j].append([count[i][j][k], float(sum(pow(embeds[count[i][j][k]]-mean_features[i][j],2)))])
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])

        for i in mean_features.keys():
            #for j in range(k):
            for j in range(len(count[i])):
                distt = np.array([dist[i][j][ind][1] for ind in range(len(dist[i][j]))])
                if one_per_class:
                    proportion=1
                else:
                    proportion = min(int(len(count[i][j]) / total_data * size), len(count[i][j]))
                ind = np.argpartition(distt, proportion)[:proportion]
                memory = [dist[i][j][idx][0] for idx in ind]
                memory = torch.from_numpy(np.array(memory))
                if replay_buffer == None:
                    replay_buffer = memory
                else:
                    replay_buffer = torch.cat((replay_buffer, memory),0)

    elif clustering == 'no':
        #각 class당 feature의 평균
        for cla in partition[task]:
            if type == 'feature':
                mean_features[cla] = torch.zeros(len(features[0])).to(device)
            elif type == 'embedding':
                mean_features[cla] = torch.zeros(embeds.size(1)).to(device)
            count[cla] = []
            dist[cla] = []

        for i in range(len(purified)):
            if type == 'feature':
                mean_features[int(labels[purified[i]])] += features[purified[i]]
            elif type == 'embedding':
                mean_features[int(labels[purified[i]])] += embeds[purified[i]]
            count[int(labels[purified[i]])].append(purified[i])

        for i in partition[task]:
            mean_features[i] /= len(count[i]) #평균값으로 환산
            for j in range(len(count[i])):
                if type == 'feature':
                    dist[i].append([count[i][j], float(sum(pow(features[count[i][j]]-mean_features[i],2)))])
                elif type == 'embedding':
                    dist[i].append([count[i][j], float(sum(pow(embeds[count[i][j]]-mean_features[i],2)))]) #dist 계산
            
        for i in mean_features.keys():
            distt = np.array([dist[i][ind][1] for ind in range(len(dist[i]))])
            if one_per_class:
                proportion = 1
            else:
                proportion = min(int(len(count[i]) / total_data * size), len(count[i]))
            ind = np.argpartition(distt, proportion)[:proportion]
            memory = [dist[i][idx][0] for idx in ind]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, mean_features, count, dist
    
def update_CM(network, type, train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance, one_per_class):
    
    if type == 'embedding':
        embeds = network.get_embeds(features, adj)[:]

    purified = {}

    for cla in partition[task]:
        cm[cla] = []
        purified[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]

    for cla in partition[task]:
        other_class = partition[task][:]
        other_class.remove(cla)
        other = []
        for clas in other_class:
            other += purified[clas]
        for idx in range(len(purified[cla])):
            if type == 'feature':
                dist = pow(features[purified[cla][idx]]-features[other],2)
            elif type == 'embedding':
                dist = pow(embeds[purified[cla][idx]]-embeds[other],2)
            counts = np.sum(dist.cpu().detach().numpy(),1)
            cm[cla].append([train_per_class[cla][idx], len(counts[counts<distance])])
    
    for i in cm.keys():
        centrality = np.array([cm[i][ind][1] for ind in range(len(cm[i]))])
        if one_per_class:
            proportion = 1
        else:
            proportion = min(int(len(train_per_class[i]) / total_data * size), len(train_per_class[i]))
        ind = np.argpartition(centrality, -proportion)[-proportion:]
        memory = [cm[i][idx][0] for idx in ind]
        memory = torch.from_numpy(np.array(memory))
        if replay_buffer == None:
            replay_buffer = memory
        else:
            replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, cm


def degree_homophily(index, adj, label):
    adj_list = adj[index].nonzero()
    degree = len(adj_list)
    count = 0
    for idx in adj_list:
        if label[index] == label[idx]:
            count += 1
    homophily = count/len(adj_list) if degree != 0 else 0

    return degree, homophily

def minmax(input):
    min_scalar = np.min(input)
    max_scalar = np.max(input)
    boundary = max_scalar - min_scalar
    temp = np.zeros(len(input))
    min = temp + min_scalar
    z = np.divide((input-min),boundary)

    return z

#def update_replay(replay, network, 'embedding', matching_index_inv, args.memory_size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, args.alpha_dist, homophily, args.beta_homo, degree, args.gamma_deg, args.one_per_class, train_per_class, args.clustering, args.k ):

def update_replay(replay_type, network, matching_index, matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, alpha, beta, gamma, one_per_class, train_per_class, clustering, k, distance, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree):
    with torch.no_grad():
        if task == len(train)-1:
            pass
        else:
            if replay_type == 'random':
                replay_buffer = None
                replay_buffer = update_random(train_per_class, matching_index_inv, partition, task, size, replay_buffer, total_data, one_per_class)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'MFf':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, 'feature', matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, one_per_class, train_per_class, clustering, k)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'MFe':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, 'embedding', matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, one_per_class, train_per_class, clustering, k)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'CMf':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, 'feature', train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance, one_per_class)
            elif replay_type == 'CMe':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, 'embedding', train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance, one_per_class)
            elif replay_type =='C_Mf':
                if task == 0:
                    cm, replay, distances, distances_mean = {}, {}, {}, {}
                replay_buffer, replay, cm, distances, distances_mean = coverage_max(network, 'feature', train_per_class, matching_index_inv, size, replay, total_data, partition, task, features, adj, cm, distance, one_per_class, distances, distances_mean)
            elif replay_type =='C_Me':
                if task == 0:
                    cm, replay, distances, distances_mean = {}, {}, {}, {}
                replay_buffer, replay, cm, distances, distances_mean = coverage_max(network, 'embedding', train_per_class, matching_index_inv, size, replay, total_data, partition, task, features, adj, cm, distance, one_per_class, distances, distances_mean)
            elif replay_type == 'MF_hd':
                if task == 0:
                    mean_features, count, dist, homophily, degree = {}, {}, {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist, homophily, degree = update_MF_hd(network, 'embedding', matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, alpha, homophily, beta, degree, gamma, one_per_class, train_per_class, clustering, k)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
    
    return replay_buffer, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree