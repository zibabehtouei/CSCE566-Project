# -*- coding: utf-8 -*-
"""
Data Mining Project
Author: Ziba Behtouei (C00549588)
Instructor: Prof. Min Shi
"""

# ==========================================================
# Professional Version: Graph Representation Learning & Node Classification
# Datasets : Cora, Citeseer
# Models   : DeepWalk, GCN
# Metrics  : Accuracy, AUC
# ==========================================================

!pip install -q gensim networkx wget

import os
import random
import tarfile
import numpy as np
import scipy.sparse as sp
import networkx as nx
import wget

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout

from gensim.models import Word2Vec

# ==========================================================
# 1. Reproducibility
# ==========================================================
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(42)

# ==========================================================
# 2. Dataset download & preparation
# ==========================================================
def _download_extract(dataset_name, url):
    if not os.path.exists(dataset_name):
        print(f"Downloading {dataset_name} dataset...")
        wget.download(url)
        with tarfile.open(f"{dataset_name}.tgz", "r:gz") as archive:
            archive.extractall()
        print(f"\n{dataset_name} ready!")

def prepare_dataset(dataset_name):
    dataset_name_lower = dataset_name.lower()
    urls = {
        "cora": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        "citeseer": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    }
    _download_extract(dataset_name_lower, urls[dataset_name_lower])

    content_path = os.path.join(dataset_name_lower, f"{dataset_name_lower}.content")
    cites_path = os.path.join(dataset_name_lower, f"{dataset_name_lower}.cites")

    # --- Load nodes and features ---
    content_data = np.genfromtxt(content_path, dtype=str)
    node_ids = content_data[:, 0]
    features = content_data[:, 1:-1].astype(np.float32)
    raw_labels = content_data[:, -1]

    labels = LabelEncoder().fit_transform(raw_labels)
    node_map = {nid: idx for idx, nid in enumerate(node_ids)}

    # --- Load edges ---
    edges_raw = np.genfromtxt(cites_path, dtype=str)
    if edges_raw.ndim == 1:
        edges_raw = np.expand_dims(edges_raw, 0)

    edges_idx = [(node_map[s], node_map[t]) for s, t in edges_raw if s in node_map and t in node_map]
    edges_idx = np.array(edges_idx)

    # --- Build adjacency matrix ---
    n_nodes = len(node_ids)
    adj = sp.coo_matrix((np.ones(len(edges_idx)), (edges_idx[:, 0], edges_idx[:, 1])), shape=(n_nodes, n_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(n_nodes)

    # --- Row-normalize features ---
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features /= row_sums

    graph_nx = nx.from_scipy_sparse_array(adj)

    return adj, features, labels, graph_nx

# ==========================================================
# 3. DeepWalk Embeddings
# ==========================================================
def perform_random_walks(graph, walk_length=40, num_walks=10):
    walks_list = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks_list.append([str(n) for n in walk])
    return walks_list

def train_deepwalk_embeddings(graph, dim=128, walk_length=40, num_walks=10, window=5, epochs=5):
    sequences = perform_random_walks(graph, walk_length, num_walks)
    w2v_model = Word2Vec(sentences=sequences, vector_size=dim, window=window, min_count=0, sg=1, workers=4, epochs=epochs)
    embeddings = np.zeros((graph.number_of_nodes(), dim), dtype=np.float32)
    for n in graph.nodes():
        embeddings[n] = w2v_model.wv[str(n)]
    return embeddings

def evaluate_node_embeddings(embeddings, labels, use_mlp=False):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, stratify=labels, random_state=42
    )
    if use_mlp:
        clf = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu", solver="adam", max_iter=500, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1500, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob, multi_class="ovr")

# ==========================================================
# 4. GCN Implementation
# ==========================================================
def normalize_adjacency(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj).dot(D_inv_sqrt).tocoo()

class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, adj_norm, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.adj_norm = tf.constant(adj_norm, dtype=tf.float32)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                      initializer="glorot_uniform", name="kernel")

    def call(self, inputs):
        support = tf.matmul(inputs, self.kernel)
        out = tf.matmul(self.adj_norm, support)
        return self.activation(out) if self.activation else out

def split_data(n_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    idx = np.arange(n_nodes)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    train_end = int(train_ratio * n_nodes)
    val_end = int((train_ratio + val_ratio) * n_nodes)
    train_mask = np.zeros(n_nodes, dtype=np.float32); train_mask[idx[:train_end]] = 1.0
    val_mask = np.zeros(n_nodes, dtype=np.float32); val_mask[idx[train_end:val_end]] = 1.0
    test_mask = np.zeros(n_nodes, dtype=np.float32); test_mask[idx[val_end:]] = 1.0
    return idx[:train_end], idx[train_end:val_end], idx[val_end:], train_mask, val_mask, test_mask

def build_gcn_model(adj_norm, n_features, n_classes, hidden_units=64, dropout=0.5, lr=0.01):
    inputs = Input(shape=(n_features,))
    x = GraphConvolutionLayer(hidden_units, adj_norm, activation="relu")(inputs)
    x = Dropout(dropout)(x)
    outputs = GraphConvolutionLayer(n_classes, adj_norm, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_evaluate_gcn(adj, features, labels, dataset_name):
    n_nodes, n_feat = features.shape
    n_classes = len(np.unique(labels))
    adj_norm = normalize_adjacency(adj).toarray()

    if dataset_name.lower() == "citeseer":
        train_ratio, val_ratio, hidden_units = 0.7, 0.1, 128
    else:
        train_ratio, val_ratio, hidden_units = 0.6, 0.2, 64

    train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = split_data(n_nodes, train_ratio, val_ratio)
    model = build_gcn_model(adj_norm, n_feat, n_classes, hidden_units)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1)
    model.fit(features, labels, sample_weight=train_mask, epochs=400, batch_size=n_nodes,
              verbose=0, validation_data=(features, labels, val_mask), callbacks=[early_stop])

    logits = model.predict(features, batch_size=n_nodes, verbose=0)
    y_pred = logits.argmax(axis=1)
    return accuracy_score(labels[test_idx], y_pred[test_idx]), roc_auc_score(labels[test_idx], logits[test_idx], multi_class="ovr")

# ==========================================================
# 5. Run experiments
# ==========================================================
final_results = []

for dataset in ["cora", "citeseer"]:
    print(f"\n======= DATASET: {dataset.upper()} =======")
    adj, features, labels, graph_nx = prepare_dataset(dataset)
    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {len(np.unique(labels))}")

    # --- DeepWalk ---
    print("[DeepWalk] Training embeddings...")
    if dataset.lower() == "citeseer":
        emb = train_deepwalk_embeddings(graph_nx, dim=256, walk_length=80, num_walks=20, epochs=10)
        acc, auc = evaluate_node_embeddings(emb, labels, use_mlp=True)
    else:
        emb = train_deepwalk_embeddings(graph_nx)
        acc, auc = evaluate_node_embeddings(emb, labels)
    print(f"[DeepWalk] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    final_results.append([dataset, "DeepWalk", acc, auc])

    # --- GCN ---
    print("[GCN] Training model...")
    acc, auc = train_and_evaluate_gcn(adj, features, labels, dataset)
    print(f"[GCN] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    final_results.append([dataset, "GCN", acc, auc])

# ==========================================================
# 6. Display results
# ==========================================================
print("\n=========== FINAL RESULTS ===========")
print("{:<10} {:<10} {:<12} {:<12}".format("Dataset","Model","Accuracy","AUC"))
for ds, model, acc, auc in final_results:
    print("{:<10} {:<10} {:<12.4f} {:<12.4f}".format(ds, model, acc, auc))