import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt
from markovchain_tk import *

from func import *

def vec_in_mat(vec, mat):
    '''
    vec(shape=(2,))是否在mat(shape=(2,n))里
    return-isin: True / False
    return-pos: 在的位置
    '''
    
    pos_row = unique_index(mat[0], vec[0])
    pos_col = unique_index(mat[1], vec[1])
    inter = set(pos_row) & set(pos_col)
    if inter:
        isin = True
        pos = inter.pop()
    else:
        isin = False
        pos = -1
    return isin, pos



# A function that plot the current graph
# A matrix should be dense.
# labels需是str类型
def plot_graph(A, labels, edge_labels=False):

    A2 = pd.DataFrame(A, index=labels, columns=labels)
    G = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, node_color='y')

    # draw edge weights
    if (edge_labels):
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.show()



def plot_graph_sparse(row_col_prob, pos_list, edge_labels=False):
    _row = row_col_prob[0]
    _col = row_col_prob[1]
    _data = row_col_prob[2]

    pos_remove_d = list(set(pos_list[0:])) #去重&排序
    labels = [str(x) for x in pos_remove_d] #转str

    A = coo_matrix((_data, (_row, _col)), shape=(len(pos_remove_d), len(pos_remove_d)))
    A_dense = A.todense()

    plot_graph(A_dense, labels, edge_labels=edge_labels)



# 通过sparse graph计算transform prob
def cal_trans_prob(graph):
    transform_prob = np.zeros(graph.shape[1])
    for i in list(set(graph[0])):
        idx = [j for j in range(graph.shape[1]) if graph[0,j]==i]
        p_sum = [graph[2,j] for j in range(graph.shape[1]) if graph[0,j]==i]
        for k in idx:
            transform_prob[k] = graph[2,k]/sum(p_sum)
    return transform_prob



def plot_graph_sparse2(row_col_prob, edge_labels=False):
    _row = row_col_prob[0]
    _col = row_col_prob[1]
    trans_prob = cal_trans_prob(row_col_prob)
    _data = np.round(trans_prob,2)

    pos_remove_d = list(set(list(_row)+list(_col))) #去重&排序
    labels = [str(x) for x in pos_remove_d] #转str

    A = coo_matrix((_data, (_row, _col)), shape=(len(pos_remove_d), len(pos_remove_d)))
    A_dense = A.todense()

    plot_graph(A_dense, labels, edge_labels=edge_labels)


def plot_graph_sparse3(row_col_prob):
    _row = row_col_prob[0]
    _col = row_col_prob[1]
    trans_prob = cal_trans_prob(row_col_prob)
    _data = np.round(trans_prob,2)

    pos_remove_d = list(set(list(_row)+list(_col))) #去重&排序
    labels = [str(x) for x in pos_remove_d] #转str

    A = coo_matrix((_data, (_row, _col)), shape=(len(pos_remove_d), len(pos_remove_d)))
    A_dense = A.todense()
    
    mc = MarkovChain(A_dense, labels, node_radius=0.4, arrow_width=0.05, arrow_head_width=0.15, fontsize=10, 
        arrow_facecolor='#696969', node_facecolor='#4682b4', node_fontsize=10)
    fig = mc.draw()


def plot_graph_sparse4(row_col_prob):
    _row = row_col_prob[0]
    _col = row_col_prob[1]
    trans_prob = cal_trans_prob(row_col_prob)
    _data = np.round(trans_prob,2)

    pos_remove_d = list(set(list(_row)+list(_col))) #去重&排序
    labels = [str(x) for x in pos_remove_d] #转str

    A = coo_matrix((_data, (_row, _col)), shape=(len(pos_remove_d), len(pos_remove_d)))
    A_dense = A.todense()
    
    mc = MarkovChain(A_dense, labels, node_radius=len(pos_remove_d)*0.15, arrow_width=0.05, arrow_head_width=len(pos_remove_d)*0.1, fontsize=10, 
        arrow_facecolor='#696969', node_facecolor='#4682b4', node_fontsize=10)
    # mc.draw()
    return mc


