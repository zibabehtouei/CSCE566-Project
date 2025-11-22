# -*- coding: utf-8 -*-
"""
Data Mining Project
Author: Ziba Behtouei (C00549588)
Instructor: Prof. Min Shi
"""

# Modified version of Graph_Project with renamed variables and rewritten comments
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

print("All modules imported successfully!")

# ---------------------------------------------
# Load dataset files
# ---------------------------------------------

# Load the node feature file for Cora
cora_data = pd.read_csv("cora.content", sep="\t", header=None)

# Node identifiers
node_list = cora_data.iloc[:, 0].values

# Feature matrix extracted from all columns except first and last
node_features = cora_data.iloc[:, 1:-1].values

# Class labels
node_labels = cora_data.iloc[:, -1].values

print("Cora content loaded!")
print("Total nodes:", len(node_list))
print("Feature size:", node_features.shape[1])

# Load citation relationships (edges)
cora_links = pd.read_csv("cora.cites", sep="\t", header=None)
edge_pairs = cora_links.values

print("Citation edges loaded:")
print("Total edges:", edge_pairs.shape[0])

# Build the initial graph
paper_graph = nx.Graph()
paper_graph.add_nodes_from(node_list)
paper_graph.add_edges_from(edge_pairs)

print("Graph constructed!")
print("Nodes in graph:", paper_graph.number_of_nodes())
print("Edges in graph:", paper_graph.number_of_edges())

# ---------------------------------------------
# Create ID-index mapping
# ---------------------------------------------
node_to_idx = {nid: i for i, nid in enumerate(node_list)}
idx_to_node = {i: nid for i, nid in enumerate(node_list)}

print("Sample node-index mapping:")
print(list(node_to_idx.items())[:5])

# Convert edge list into index-based edges
indexed_edge_list = []
for a, b in edge_pairs:
    if a in node_to_idx and b in node_to_idx:
        indexed_edge_list.append((node_to_idx[a], node_to_idx[b]))

print("Index-based edges:", len(indexed_edge_list))

# Construct index-based graph
G_idx = nx.Graph()
G_idx.add_nodes_from(range(len(node_list)))
G_idx.add_edges_from(indexed_edge_list)

print("Indexed graph created!")
print("Nodes:", G_idx.number_of_nodes())
print("Edges:", G_idx.number_of_edges())
!pip install gensim
# ---------------------------------------------
# DeepWalk Random Walks
# ---------------------------------------------
from gensim.models import Word2Vec
import random

# Function to generate random walks
def gen_random_walk(graph, start, length=40):
    route = [start]
    curr = start
    for _ in range(length - 1):
        nbrs = list(graph.neighbors(curr))
        if not nbrs:
            break
        curr = random.choice(nbrs)
        route.append(curr)
    return route

walks_per_node = 10
walk_len = 40
all_paths = []
all_nodes = list(G_idx.nodes())
print("Total nodes:", len(all_nodes))

for _ in range(walks_per_node):
    random.shuffle(all_nodes)
    for nd in all_nodes:
        path = gen_random_walk(G_idx, nd, length=walk_len)
        all_paths.append([str(x) for x in path])

print("Generated walks:", len(all_paths))
print("Example walk:", all_paths[0][:10])

# ---------------------------------------------
# Train Word2Vec for node embeddings
# ---------------------------------------------
embed_size = 128

w2v = Word2Vec(
    sentences=all_paths,
    vector_size=embed_size,
    window=5,
    min_count=0,
    sg=1,
    workers=4,
    epochs=5
)

print("Word2Vec training completed!")

# Extract embeddings
N = G_idx.number_of_nodes()
emb_matrix = np.zeros((N, embed_size))

for i in range(N):
    emb_matrix[i] = w2v.wv[str(i)]

print("Embedding matrix shape:", emb_matrix.shape)

# ---------------------------------------------
# Encode labels
# ---------------------------------------------
label_classes = sorted(list(set(node_labels)))
label_to_id = {c: i for i, c in enumerate(label_classes)}

y_data = np.array([label_to_id[l] for l in node_labels])
num_classes = len(label_classes)

print("Classes:", num_classes)
print("Label mapping sample:", list(label_to_id.items())[:5])

# ---------------------------------------------
# Train-test split for DeepWalk classification
# ---------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    emb_matrix, y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Train logistic regression on embeddings
classifier = LogisticRegression(max_iter=1000, multi_class='ovr')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

acc_val = accuracy_score(y_test, y_pred)
auc_val = roc_auc_score(label_binarize(y_test, classes=np.arange(num_classes)), y_prob, average='macro')

print("DeepWalk Accuracy:", acc_val)
print("DeepWalk AUC:", auc_val)

# ---------------------------------------------
# Prepare for GCN
# ---------------------------------------------
from sklearn.model_selection import train_test_split

all_idx = np.arange(N)

# 60% train, 20% val, 20% test
i_train, i_temp, y_train_temp, y_temp = train_test_split(
    all_idx, y_data, test_size=0.4, stratify=y_data, random_state=42
)

i_val, i_test, y_val_temp, y_test_temp = train_test_split(
    i_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train nodes:", len(i_train))
print("Val nodes:", len(i_val))
print("Test nodes:", len(i_test))

# ---------------------------------------------
# Build normalized adjacency
# ---------------------------------------------
import torch

A = nx.to_numpy_array(G_idx)
A = A + np.eye(N)

deg = A.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt

X_tensor = torch.tensor(node_features, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.long)
A_tensor = torch.tensor(A_hat, dtype=torch.float32)

t_train = torch.tensor(i_train, dtype=torch.long)
t_val = torch.tensor(i_val, dtype=torch.long)
t_test = torch.tensor(i_test, dtype=torch.long)

# ---------------------------------------------
# Define GCN
# ---------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, X, A_norm):
        return self.lin(torch.matmul(A_norm, X))

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.5):
        super().__init__()
        self.layer1 = SimpleGCNLayer(in_dim, hidden_dim)
        self.layer2 = SimpleGCNLayer(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, X, A_norm):
        h = F.relu(self.layer1(X, A_norm))
        h = self.drop(h)
        return self.layer2(h, A_norm)

model_gcn = SimpleGCN(X_tensor.shape[1], 64, num_classes, drop=0.5)
opt = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

# ---------------------------------------------
# Train GCN
# ---------------------------------------------
def eval_accuracy(model, X, A_norm, idx, labels):
    model.eval()
    with torch.no_grad():
        out = model(X, A_norm)
        preds = torch.argmax(out[idx], dim=1)
        return (preds == labels[idx]).float().mean().item()

for ep in range(200):
    model_gcn.train()
    opt.zero_grad()

    out = model_gcn(X_tensor, A_tensor)
    loss = loss_fn(out[t_train], y_tensor[t_train])

    loss.backward()
    opt.step()

    if (ep + 1) % 20 == 0:
        tr_acc = eval_accuracy(model_gcn, X_tensor, A_tensor, t_train, y_tensor)
        vl_acc = eval_accuracy(model_gcn, X_tensor, A_tensor, t_val, y_tensor)
        print(f"Epoch {ep+1:03d} | Loss {loss.item():.4f} | Train Acc {tr_acc:.4f} | Val Acc {vl_acc:.4f}")

# ---------------------------------------------
# Evaluate on test
# ---------------------------------------------
model_gcn.eval()
with torch.no_grad():
    logits = model_gcn(X_tensor, A_tensor)
    probas = F.softmax(logits, dim=1)

y_true_t = y_tensor[t_test].numpy()
y_prob_t = probas[t_test].numpy()
y_pred_t = np.argmax(y_prob_t, axis=1)

acc_gcn = accuracy_score(y_true_t, y_pred_t)
auc_gcn = roc_auc_score(label_binarize(y_true_t, classes=np.arange(num_classes)), y_prob_t, average="macro")

print("GCN Test Accuracy:", acc_gcn)
print("GCN Test AUC:", auc_gcn)