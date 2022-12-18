import torch
import numpy as np 
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import CitationFull
import collections
import random
from linkprediction import Net
import os.path as osp
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import torch_geometric.transforms as T
import torch.nn.functional as F

def load_data(args, dataset="cora", classes_per_task=2):
    
    print("Dataset : {} ". format(dataset))

    if dataset == 'cora' or dataset == 'citeseer':

        path="./data/"+dataset+"/" 

        idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:,-1])
        num_classes = labels.shape[1]

        idx = np.array(idx_features_labels[:,0],dtype=np.dtype(str))
        idx_map = {j: i for i,j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path,dataset), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.dtype(str)).reshape(edges_unordered.shape)
        #indexx = list(range(len(idx)))
        #idx_train, idx_test = train_test_split(indexx, test_size = 0.3, random_state= 1)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
        features = normalize_features(features)
        adj = normalize_adj(adj+sp.eye(adj.shape[0]))
        
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        labels_np = labels.numpy()
        label_counter = collections.Counter(labels_np)
        selected_ids = [id for id, count in label_counter.items() if count > 200]
        index = list(range(len(labels)))

        selected_ids = sorted(selected_ids)
        handle = [item for item in index if labels[item] in selected_ids]

        device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')
        adj = adj_masking(adj, handle, device)
        features = feature_masking(features, handle)
        labels = label_masking(labels, handle)
        index = list(range(len(labels)))

        sorted_ids = sorted(selected_ids)
        #sorted_ids = [2,3,4,1,0,6]  #order change

        for idx in index:
            labels[idx] = torch.tensor(sorted_ids.index(labels[idx]))

        selected_ids = list(range(len(selected_ids)))

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        total_classes = int(max(labels))+1
        num_classes = len(selected_ids)

        idx_train, idx_test = train_test_split(index, test_size = 0.3, random_state=1)

    elif dataset == 'corafull':
        
        Data = CitationFull('./data', 'Cora')

        index = list(range(len(Data.data.x)))
        
        edges = Data.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])), shape=(num_nodes, num_nodes))

        adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
        adj = normalize_adj(adj+sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))

        features = Data.data.x
        labels = Data.data.y

        labels_np = labels.numpy()
        label_counter = collections.Counter(labels_np)
        selected_ids = [id for id, count in label_counter.items() if count > 530]

        selected_ids = sorted(selected_ids)
        handle = [item for item in index if labels[item] in selected_ids]

        device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')
        adj = adj_masking(adj, handle, device)
        features = feature_masking(features, handle)
        labels = label_masking(labels, handle)
        index = list(range(len(labels)))

        sorted_ids = sorted(selected_ids)

        for idx in index:
            labels[idx] = torch.tensor(sorted_ids.index(labels[idx]))

        selected_ids = list(range(len(selected_ids)))

        #random.seed(args.seed_shuffle)
        #random.shuffle(selected_ids)

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        total_classes = int(max(labels))+1
        num_classes = len(selected_ids)

        idx_train, idx_test = train_test_split(index, test_size = 0.3, random_state=1)

    idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    train = {}
    partition = {}


    class_per_task = classes_per_task
    num_tasks = len(selected_ids) // class_per_task
    for i in range(num_tasks):
        train[i] = []
        partition[i] = []

    train_per_class = {}
    test_per_class = {}
    class_idx = {}
    for i in range(len(selected_ids)):
        train_per_class[selected_ids[i]] = []
        test_per_class[selected_ids[i]] = []
        class_idx[selected_ids[i]] = []


    for i in range(len(idx_train)):
        if int(labels[idx_train[i]]) in selected_ids:
            if selected_ids.index(labels[idx_train[i]]) // class_per_task < num_tasks:
                train[selected_ids.index(labels[idx_train[i]])//class_per_task].append(idx_train[i])
            else:
                train[selected_ids.index(labels[idx_train[i]])//class_per_task-1].append(idx_train[i])
            train_per_class[int(labels[idx_train[i]])].append(idx_train[i])
            class_idx[int(labels[idx_train[i]])].append(idx_train[i])

    for i in range(len(selected_ids)):
        if i // class_per_task < num_tasks:
            partition[i//class_per_task].append(selected_ids[i])
        else:
            partition[i//class_per_task-1].append(selected_ids[i])
    

    for j in range(len(idx_test)):
        if int(labels[idx_test[j]]) in selected_ids:
            test_per_class[int(labels[idx_test[j]])].append(idx_test[j])
            class_idx[int(labels[idx_test[j]])].append(idx_test[j])

    for i in range(num_tasks):
        train[i] = torch.LongTensor(train[i])
    

    return adj, features, labels, train, total_classes, partition, train_per_class, test_per_class, class_idx


def adj_masking(adj, handled, device):
    adj = adj[handled,:]
    adj = adj[:,handled]

    A_tilde = adj.to(device) + torch.eye(adj.size(0)).to(device)
    D_tilde_inv_sqrt = torch.diag(torch.sqrt(torch.sum(A_tilde, dim = 1)) ** -1)
    adj = torch.mm(D_tilde_inv_sqrt, torch.mm(A_tilde, D_tilde_inv_sqrt)).to(device)
    return adj

def feature_masking(features, handled):
    features = features[handled, :]
    return features

def label_masking(labels, handled):
    labels = labels[handled]
    return labels

def zscore(input):
    mean_scalar = np.mean(input)
    std_scalar = np.std(input)
    temp = np.zeros(len(input))
    mean = temp + mean_scalar
    std = temp + std_scalar
    z = np.divide((input-mean),std)

    return z


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def task_accuracy(task, output, labels, class_per_task):
    preds = output[:,class_per_task*task:class_per_task*(task+1)].max(1)[1].type_as(labels)
    preds = torch.add(preds, class_per_task*task)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def normalize_adj(mx): # A_hat = DAD
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx_to =  mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx_to

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_to =  r_mat_inv.dot(mx) 
    return mx_to 

def encode_onehot(labels):
    classes = sorted(set(labels))
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def adj_to_edge_index(adj, device, edge_index=np.empty((2,1),int)):
    for i in range(len(adj[0])):
        adj_temp = adj[i].cpu().numpy()
        indexes = np.where(adj_temp>0)
        for idx in indexes[0]:
            edge_index = np.append(edge_index, [[i],[idx]], axis=1)
    edge_index = torch.from_numpy(edge_index)
    edge_index = np.delete(edge_index, [0], 1)
    return edge_index.to(device)

def edge_index_to_adj(edge_index, dim):
    adj = torch.zeros(dim, dim)
    for i in range(len(edge_index[0])):
        adj[edge_index[0][i]][edge_index[1][i]] = 1
        adj[edge_index[1][i]][edge_index[0][i]] = 1
    adj = normalize_adj_for_updated(adj)
    return adj

def normalize_adj_for_updated(mx): # A_hat = DAD
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    mx_to = np.matmul(np.matmul(mx.numpy(), r_mat_inv_sqrt).transpose() , r_mat_inv_sqrt)
    return torch.Tensor(mx_to)

def train_linkpred(features, adj, device):
    dataset_lp = 'Cora'
    path = osp.join('../','data',dataset_lp)
    dataset_lp = Planetoid(path, dataset_lp, transform=T.NormalizeFeatures())
    data = dataset_lp[0]
    data.x = features
    edge_index = adj_to_edge_index(adj, device, edge_index=np.empty((2,1),int))
    data.edge_index = edge_index
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    model_lp, data = Net().to(device), data.to(device)
    optimizer_lp = torch.optim.Adam(params=model_lp.parameters(), lr=0.01)
    neg_edge_index = negative_sampling(
                    edge_index=data.train_pos_edge_index, # positive edges
                    num_nodes = data.num_nodes, # number of nodes
                    num_neg_samples = data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
    link_logits = get_link_logits(model_lp, data.x, data.train_pos_edge_index, neg_edge_index)
    link_logits.sigmoid()
    train_lp(model_lp, data, optimizer_lp, device)
    best_val_perf = test_perf = 0
    for epoch_lp in range(1,301):
        train_loss_lp = train_lp(model_lp, data, optimizer_lp, device)
        val_perf, tmp_test_perf = test_lp(model_lp, data, device)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        # log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # if epoch_lp % 10 == 0:
            # print(log.format(epoch_lp, train_loss_lp, best_val_perf, test_perf))
    z = model_lp.encode(features, edge_index)
    final_edge_index = model_lp.decode_all(z)
    dim = len(adj)
    updated_adj = edge_index_to_adj(final_edge_index, dim)
    return model_lp, updated_adj.to(device)
    
def update_adj_with_linkpred(features, adj, model_lp, device):
    edge_index = adj_to_edge_index(adj, device, edge_index=np.empty((2,1),int))
    z = model_lp.encode(features, edge_index)
    final_edge_index = model_lp.decode_all(z)
    dim = len(adj)
    updated_adj = edge_index_to_adj(final_edge_index, dim)
    return updated_adj.to(device)

def get_link_logits(model_lp, x, edge_index, neg_edge_index):
    z = model_lp.encode(x, edge_index) # encode
    link_logits = model_lp.decode(z, edge_index, neg_edge_index) # decode
    return link_logits

def get_link_labels(pos_edge_index, neg_edge_index, device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train_lp(model_lp, data, optimizer_lp, device):
    model_lp.train()
    neg_edge_index = negative_sampling(
                    edge_index = data.train_pos_edge_index,
                    num_nodes = data.num_nodes,
                    num_neg_samples = data.train_pos_edge_index.size(1))
    link_logits = get_link_logits(model_lp, data.x, data.train_pos_edge_index, neg_edge_index)
    optimizer_lp.zero_grad()
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer_lp.step()
    return loss

def test_lp(model_lp, data, device):
    with torch.no_grad():
        model_lp.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index = data[f'{prefix}_pos_edge_index']
            neg_edge_index = data[f'{prefix}_neg_edge_index']
            link_logits = get_link_logits(model_lp, data.x, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)
            perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        return perfs
