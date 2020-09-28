import pandas as pd
import numpy as np
import math
import networkx as nx
import os
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap
from scipy.stats import spearmanr, kendalltau, wilcoxon, mannwhitneyu
import topcorr


/home/tristan/Documents/phd/financial_network_clustering/mst_difference.png
/home/tristan/Documents/phd/financial_network_clustering/total_prescence_corr.png
/home/tristan/Documents/phd/financial_network_clustering/total_prescence_partial_correlation.png
/home/tristan/Documents/phd/financial_network_clustering/total_presence_correlation.png
/home/tristan/Documents/phd/financial_network_clustering/total_presence_partial_correlation.png
/home/tristan/Documents/phd/financial_network_clustering/us_average_path_length.png
/home/tristan/Documents/phd/financial_network_clustering/us_clustering_coefficient.png
/home/tristan/Documents/phd/financial_network_clustering/us_correlation_mst.png
def find_edge_p_values(edges, p_values_pearson, p_values_spearman, p_values_tau):
    for edge in edges:
        ind1 = company_names == edge[0]
        ind2 = company_names == edge[1]
        print(edge)
        print(p_values_pearson[ind1, ind2])
        print(p_values_spearman[ind1, ind2])
        print(p_values_tau[ind1, ind2])

def calculate_diff(mst_lst):
    num_msts = len(mst_lst)
    diffs = []
    p = mst_lst[0].shape[0]
    for i in range(num_msts):
        for j in range(i+1, num_msts):
            diffs.append(np.count_nonzero(mst_lst[i] - mst_lst[j]) / (4*(p-1)))
    return diffs

def calculate_full_matrix_diff(mat_lst):
    num_mats = len(mat_lst)
    diffs = []
    p = mat_lst[0].shape[0]
    for i in range(num_mats):
        for j in range(i+1, num_mats):
            diffs.append(np.abs(mat_lst[i]/mat_lst[i].sum() - mat_lst[j]/mat_lst[j].sum()).sum())
    return diffs

def find_mean_p_values(p_values, company_sectors, M):
    """
    Measures the relationship between p-value and sector membership by 
    looking at the mean p-value for intra and inter sector edges
    """
    p = p_values.shape[0]
    intra_sector_score = []
    inter_sector_score = []
    all_score = []
    for i in range(p):
        for j in range(p):
            if M[i, j] == 0:
                continue
            all_score.append(p_values[i, j])
            if company_sectors[i] == company_sectors[j]:
                intra_sector_score.append(p_values[i, j])
            else:
                inter_sector_score.append(p_values[i, j])

    return all_score, inter_sector_score, intra_sector_score

def calculate_tau_matrix(X):
    n, p = X.shape

    M = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            if i == j:
                continue
            M[i, j], _ = kendalltau(X[:, i].reshape(-1, 1), X[:, j])

    return M + M.T
np.seterr(all='raise')

country = "DE"
if country == "DE":
    df = pd.read_csv("DAX30.csv", index_col=0)        
    window_size = 252 * 4
    bootstrap_size = 252
elif country == "UK":
    df = pd.read_csv("FTSE100.csv", index_col=0)
    window_size = 252 * 4
    bootstrap_size = 252
elif country == "US":
    df = pd.read_csv("S&P500.csv", index_col=0)
    window_size = 252 * 4
    bootstrap_size = 252

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

num_removal_runs = 1000
no_samples = X.shape[0]
p = X.shape[1]

X_new = X[0:window_size, 0:70]
company_names = company_names[0:70]
company_sectors = company_sectors[0:70]

p = X_new.shape[1]
bs = CircularBlockBootstrap(bootstrap_size, X_new)
total_mst_prescence_spearman = np.zeros((p, p))
total_mst_prescence_pearson = np.zeros((p, p))
total_mst_prescence_tau = np.zeros((p, p))
pearson_msts = []
spearman_msts = []
tau_msts = []

pearson_full = []
spearman_full = []
tau_full = []
i = 0
for data in bs.bootstrap(num_removal_runs):
    print("Run %s" % i)
    X_bs = data[0][0]

    pearson_corr = np.corrcoef(X_bs.T)
    spearman_corr, _ = spearmanr(X_bs)   
    tau_corr = calculate_tau_matrix(X_bs)

    pearson_full.append(pearson_corr)
    spearman_full.append(spearman_corr)
    tau_full.append(tau_corr)

    mst_pearson = topcorr.mst(pearson_corr)
    mst_spearman = topcorr.mst(spearman_corr)
    mst_tau = topcorr.mst(tau_corr)

    # Build a relabelling dictionary
    node_labels = dict()
    for node in mst_pearson.nodes():
        node_labels[node] = company_names[node]

    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))

    mst_pearson=nx.relabel_nodes(mst_pearson, node_labels)
    nx.set_node_attributes(mst_pearson, node_attributes, 'sector')

    mst_spearman=nx.relabel_nodes(mst_spearman, node_labels)
    nx.set_node_attributes(mst_spearman, node_attributes, 'sector')
    
    mst_tau = nx.relabel_nodes(mst_tau, node_labels)
    nx.set_node_attributes(mst_tau, node_attributes, 'sector')

    M = nx.to_numpy_array(mst_pearson, nodelist=company_names)
    pearson_msts.append(M)
    total_mst_prescence_pearson += M 

    M = nx.to_numpy_array(mst_spearman, nodelist = company_names)
    spearman_msts.append(M)
    total_mst_prescence_spearman += M 
    
    M = nx.to_numpy_array(mst_tau, nodelist = company_names)
    tau_msts.append(M)
    total_mst_prescence_tau += M 
    i+=1

pearson_weighted_diffs = calculate_full_matrix_diff(pearson_msts)
spearman_weighted_diffs = calculate_full_matrix_diff(spearman_msts)
tau_weighted_diffs = calculate_full_matrix_diff(tau_msts)

pearson_unweighted_diffs = calculate_diff(pearson_msts)
spearman_unweighted_diffs = calculate_diff(spearman_msts)
tau_unweighted_diffs = calculate_diff(tau_msts)

# Full correlation matrices stdev
pearson_full_diff = calculate_full_matrix_diff(pearson_full)
spearman_full_diff = calculate_full_matrix_diff(spearman_full)
tau_full_diff = calculate_full_matrix_diff(tau_full)

print("Pearson & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (np.mean(pearson_weighted_diffs), np.std(pearson_weighted_diffs), np.mean(pearson_unweighted_diffs), np.std(pearson_unweighted_diffs), np.mean(pearson_full_diff), np.std(pearson_full_diff)))
print("Spearman & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (np.mean(spearman_weighted_diffs), np.std(spearman_weighted_diffs), np.mean(spearman_unweighted_diffs), np.std(spearman_unweighted_diffs), np.mean(spearman_full_diff), np.std(spearman_full_diff)))
print("$\\tau$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f  \\\\" % (np.mean(tau_weighted_diffs), np.std(tau_weighted_diffs), np.mean(tau_unweighted_diffs), np.std(tau_unweighted_diffs), np.mean(tau_full_diff), np.std(tau_full_diff)))


plt.show()