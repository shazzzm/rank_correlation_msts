import pandas as pd
import numpy as np
import math
import networkx as nx
import os
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap
from scipy.stats import spearmanr, kendalltau

def correlation_to_distance_graph(G):
    """
    Converts a correlation graph to a distance based one
    """
    G = G.copy()
    for edge in G.edges():
        G.edges[edge]['weight'] =  np.sqrt(2 - 2*G.edges[edge]['weight'])
    return G

def calculate_diff(mst_lst):
    num_msts = len(mst_lst)
    diffs = []
    p = mst_lst[0].shape[0]
    for i in range(num_msts):
        for j in range(i+1, num_msts):
            diffs.append(np.count_nonzero(mst_lst[i] - mst_lst[j]) / (4*(p-1)))
    return np.mean(diffs), np.std(diffs)

def compute_PMFG(sorted_edges, nb_nodes):
    PMFG = nx.Graph()
    for edge in sorted_edges:
        PMFG.add_edge(edge['source'], edge['dest'])
        if not planarity.is_planar(PMFG):
            PMFG.remove_edge(edge['source'], edge['dest'])
            
        if len(PMFG.edges()) == 3*(nb_nodes-2):
            break
    
    return PMFG

def sort_graph_edges(G):
    sorted_edges = []
    for source, dest, data in sorted(G.edges(data=True),
                                     key=lambda x: x[2]['weight']):
        sorted_edges.append({'source': source,
                             'dest': dest,
                             'weight': data['weight']})
        
    return sorted_edges

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

    return np.mean(all_score), np.std(all_score), np.mean(intra_sector_score), np.std(intra_sector_score), np.mean(inter_sector_score), np.std(inter_sector_score)

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

df = pd.read_csv("S&P500.csv", index_col=0)

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

window_size = 252*3
slide_size = 30
bootstrap_size = 252*2
num_removal_runs = 100
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
print("We're running %s times" % no_runs)

X_new = X[0:window_size, :]
bs = CircularBlockBootstrap(bootstrap_size, X_new)
total_mst_prescence_spearman = np.zeros((p, p))
total_mst_prescence_pearson = np.zeros((p, p))
total_mst_prescence_tau = np.zeros((p, p))
pearson_msts = []
spearman_msts = []
tau_msts = []

for data in bs.bootstrap(num_removal_runs):
    X_bs = data[0][0]

    corr = np.corrcoef(X_bs.T)

    G_pearson = nx.from_numpy_matrix(corr)
    G_pearson = nx.relabel_nodes(G_pearson, dict(zip(G_pearson.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G_pearson, node_attributes, 'sector')

    corr, p = spearmanr(X_bs)   
    G_spearman = nx.from_numpy_matrix(corr)
    G_spearman = nx.relabel_nodes(G_spearman, dict(zip(G_spearman.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G_spearman, node_attributes, 'sector')

    corr = calculate_tau_matrix(X_bs)   
    G_tau = nx.from_numpy_matrix(corr)
    G_tau = nx.relabel_nodes(G_tau, dict(zip(G_tau.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G_tau, node_attributes, 'sector')

    G_pearson_dist = correlation_to_distance_graph(G_pearson)
    G_spearman_dist = correlation_to_distance_graph(G_spearman)
    G_tau_dist = correlation_to_distance_graph(G_tau)

    G_pearson_mst = nx.minimum_spanning_tree(G_pearson_dist)
    G_spearman_mst = nx.minimum_spanning_tree(G_spearman_dist)
    G_tau_mst = nx.minimum_spanning_tree(G_tau_dist)

    M = nx.to_numpy_array(G_pearson_mst, weight='None')
    pearson_msts.append(M)
    total_mst_prescence_pearson += M 

    M = nx.to_numpy_array(G_spearman_mst, weight='None')
    spearman_msts.append(M)
    total_mst_prescence_spearman += M 
    
    M = nx.to_numpy_array(G_tau_mst, weight='None')
    tau_msts.append(M)
    total_mst_prescence_tau += M 

print("Pearson")
print("%s \pm %s" % calculate_diff(pearson_msts))

print("Spearman")
print("%s \pm %s" % calculate_diff(spearman_msts))

print("tau")
print("%s \pm %s" % calculate_diff(tau_msts))

plt.figure()
plt.title("Pearson Correlation")
plt.hist(total_mst_prescence_pearson)
plt.savefig("mst_prescence_pearson.png")

plt.figure()
plt.title("Spearman Correlation")
plt.hist(total_mst_prescence_spearman)
plt.savefig("mst_prescence_spearman.png")

pearson_corr = np.corrcoef(X_new.T)
spearman_corr, p = spearmanr(X_new)   
kendall_corr = calculate_tau_matrix(X_new)

np.fill_diagonal(pearson_corr, 0)
np.fill_diagonal(spearman_corr, 0)

p_values_pearson = total_mst_prescence_pearson / num_removal_runs
p_values_spearman = total_mst_prescence_spearman / num_removal_runs
p_values_tau = total_mst_prescence_tau / num_removal_runs

np.fill_diagonal(p_values_pearson, 0)
np.fill_diagonal(p_values_spearman, 0)

plt.figure()
plt.scatter(pearson_corr.flatten(), p_values_pearson.flatten())
#plt.title("Correlation strength against p-value")
plt.xlabel("Pearson Correlation Strength")
plt.ylabel("p value")
plt.xlim((-0.2, 1))
plt.savefig("pvalue_correlation_pearson.png")
print("Correlation between correlation and p-value")
print(spearmanr(pearson_corr.flatten(), p_values_pearson.flatten()))

plt.figure()
plt.scatter(spearman_corr.flatten(), p_values_spearman.flatten())
#plt.title("Partial Correlation strength against p-value")
plt.xlabel("Spearman Correlation Strength")
plt.ylabel("p value")
plt.xlim((-0.2, 1))
plt.savefig("pvalue_correlation_spearman.png")

print("Correlation between spearman correlation and p-value")
print(spearmanr(spearman_corr.flatten(), p_values_spearman.flatten()))

plt.figure()
plt.scatter(kendall_corr.flatten(), p_values_tau.flatten())
#plt.title("Partial Correlation strength against p-value")
plt.xlabel("Spearman Correlation Strength")
plt.ylabel("p value")
plt.xlim((-0.2, 1))
plt.savefig("pvalue_correlation_tau.png")

print("Correlation between tai correlation and p-value")
print(spearmanr(kendall_corr.flatten(), p_values_tau.flatten()))

corr = np.corrcoef(X_new.T)

G_pearson = nx.from_numpy_matrix(corr)
G_pearson = nx.relabel_nodes(G_pearson, dict(zip(G_pearson.nodes(), company_names)))
node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
nx.set_node_attributes(G_pearson, node_attributes, 'sector')

corr, _ = spearmanr(X_new)   
G_spearman = nx.from_numpy_matrix(corr)
G_spearman = nx.relabel_nodes(G_spearman, dict(zip(G_spearman.nodes(), company_names)))
node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
nx.set_node_attributes(G_spearman, node_attributes, 'sector')

corr = calculate_tau_matrix(X_new)   
G_tau = nx.from_numpy_matrix(corr)
G_tau = nx.relabel_nodes(G_tau, dict(zip(G_tau.nodes(), company_names)))
node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
nx.set_node_attributes(G_tau, node_attributes, 'sector')

G_pearson_dist = correlation_to_distance_graph(G_pearson)
G_spearman_dist = correlation_to_distance_graph(G_spearman)
G_tau_dist = correlation_to_distance_graph(G_tau)

G_pearson_mst = nx.minimum_spanning_tree(G_pearson_dist)
G_spearman_dist = nx.minimum_spanning_tree(G_spearman_dist)
G_tau_dist = nx.minimum_spanning_tree(G_tau_dist)

M_pearson_mst = nx.to_numpy_array(G_pearson_mst)
M_pearson_mst[np.abs(M_pearson_mst) > 0] = 1

M_spearman_mst = nx.to_numpy_array(G_spearman_dist)
M_spearman_mst[np.abs(M_spearman_mst) > 0] = 1

M_tau_mst = nx.to_numpy_array(G_tau_dist)
M_tau_mst[np.abs(M_tau_mst) > 0] = 1

print("P-value scoring")
print("Pearson")
print(find_mean_p_values(p_values_pearson, company_sectors, M_pearson_mst))

print("Spearman")
print(find_mean_p_values(p_values_spearman, company_sectors, M_spearman_mst))

print("Tau")
print(find_mean_p_values(p_values_tau, company_sectors, M_tau_mst))

# Most probable edges are
print("Most probable edges:")
ind = np.argsort(p_values_pearson, axis=None)[::-1]
cutoff_val = p_values_pearson.flatten()[ind[20]]
p_values_pearson[cutoff_val > p_values_pearson] = 0
p_values_pearson[p_values_pearson > 0] = 1
nonzero_ind = np.nonzero(p_values_pearson)
x_cords = nonzero_ind[0]
y_cords = nonzero_ind[1]
for x in zip(x_cords, y_cords):
    print("%s - %s" % (company_names[x[0]], company_names[x[1]]))
    print("%s - %s" % (company_sectors[x[0]], company_sectors[x[1]]))
    print()

plt.show()