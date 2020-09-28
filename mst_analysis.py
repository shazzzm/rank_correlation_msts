import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
import os
import pandas as pd
from pathlib import Path
import operator
import matplotlib
from scipy.stats import spearmanr, pearsonr, kendalltau
import topcorr
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import QuantileTransformer
import cvxpy as cvx

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

def mean_occupation_layer(mst_G):
    """
    Calculates the mean occupation layer, uses the node with the highest degree
    as the center of the tree
    """
    central_node = sorted(mst_G.degree(), key=lambda x: x[1], reverse=True)[0][0]
    shortest_paths = nx.shortest_path_length(mst_G, source=central_node)
    mean_ol = np.array(list(shortest_paths.values())).mean()
    return mean_ol

def measure_leaf_fraction(mst_G):
    """
    Calculates the fraction of edges with degree 1
    """
    M = nx.to_numpy_array(mst_G)
    p = M.shape[0]
    M[np.abs(M) > 0] = 1
    degree = M.sum(axis=0)

    return np.count_nonzero(degree == 1)/(2*(p-1))

def sort_dict(dct):
    """
    Takes a dict and returns a sorted list of key value pairs
    """
    sorted_x = sorted(dct.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_x

def calculate_mst_diff(mst, prev_mst):
    nodes = list(mst.nodes)
    M = nx.to_numpy_array(mst, nodes)
    p = M.shape[0]
    M_prev = nx.to_numpy_array(prev_mst, nodes)
    weighted_node_diff = np.abs(M/M.sum() - M_prev/M_prev.sum()).sum(axis=0)
    M[np.abs(M) > 0] = 1
    M_prev[np.abs(M_prev) > 0] = 1

    fraction_total_diff = np.count_nonzero(M - M_prev)/(4*(p-1))
    node_degree = M.sum(axis=0)/M.sum()
    node_degree_2 = M_prev.sum(axis=0)/M.sum()

    M_diff = M - M_prev
    return fraction_total_diff, weighted_node_diff

def correlation_to_distance(G):
    """
    Converts a correlation graph to a distance based one
    """
    G = G.copy()
    for edge in G.edges():
        G.edges[edge]['weight'] =  np.sqrt(2 - 2*G.edges[edge]['weight'])
    return G

def calculate_exponent(G):
    """
    Calculates the power law exponent for the degree distribution
    """
    M = nx.to_numpy_array(G)
    p = M.shape[0]
    M[np.abs(M) > 0] = 1
    degrees = M.sum(axis=0)

    alpha = 1 + p * np.reciprocal(np.log(degrees).sum())
    return alpha

def calculate_tau_matrix(X):
    n, p = X.shape

    M = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            if i == j:
                continue
            M[i, j], _ = kendalltau(X[:, i].reshape(-1, 1), X[:, j])

    return M + M.T

def get_centrality(G, degree=True):
    """
    Calculates the centrality of each node and mean centrality of a sector 
    if degree is true we use degree centrality, if not we use betweeness centrality
    """
    node_centrality = collections.defaultdict(float)
    total = 0

    if not degree:
        # Do betweeness centrality
        node_centrality = nx.betweenness_centrality(G)
    else:
        # Calculate the edge centrality
        node_centrality = nx.degree_centrality(G)

    sorted_centrality = sort_dict(node_centrality)
    centrality_names = [x[0] for x in sorted_centrality]
    centrality_sectors = []

    for name in centrality_names:
        centrality_sectors.append(G.nodes[name]['sector'])

    # Figure out the mean centrality of a sector
    sector_centrality = collections.defaultdict(float)
    no_companies_in_sector = collections.defaultdict(int)

    for comp in G:
        sector = G.nodes[comp]['sector']
        sector_centrality[sector] += node_centrality[comp]
        no_companies_in_sector[sector] += 1
    for sec in sector_centrality:
        sector_centrality[sec] /= no_companies_in_sector[sec]

    # Ensure these add to one
    s = 0
    for sec in sector_centrality:
        s += sector_centrality[sec]

    for sec in sector_centrality:
        sector_centrality[sec] /= s
    return node_centrality, sector_centrality

def calculate_min_risk(covariance, means, verbose=False, diversification_value=1, enable_short_selling=False):
    w = cvx.Variable(covariance.shape[0])
    gamma = cvx.Parameter(nonneg=True)
    risk = cvx.quad_form(w, covariance)
    ret = means.T @ w
    if enable_short_selling:
        w_min = -np.Inf
    else:
        w_min = 0
    prob = cvx.Problem(cvx.Minimize(risk), 
                [cvx.sum(w) == 1, 
                w >= w_min, diversification_value >= w])
    prob.solve(solver=cvx.CVXOPT, verbose=verbose)
    risk_data = cvx.sqrt(risk).value
    ret_data = ret.value 
    w = w.value
    return _, _, w


plt.rcParams.update({'figure.max_open_warning': 0})

# If selected this will use the quantile transformer
make_normal = False

# Set the country you desire to analyze
country = "DE"
if country == "DE":
    df = pd.read_csv("DAX30.csv", index_col=0)        
    window_size = 252 * 2
    #window_size = 126 
elif country == "UK":
    df = pd.read_csv("FTSE100.csv", index_col=0)
    window_size = 252 * 2
    # window_size = 126 
elif country == "US":
    df = pd.read_csv("S&P500.csv", index_col=0)
    window_size = 252 * 2
    #window_size = 126 

#window_size = 40

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sector_set = sorted(set(company_sectors))

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2.index = pd.to_datetime(df_2.index)

df_2 = np.log(df_2) - np.log(df_2.shift(1))

X = df_2.values[1:, :]

slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])
dt = pd.to_datetime(dates)
dt_2 = dt[1:]

corrs_pearson = []
corrs_spearman = []
corrs_tau = []

pearson_largest_eig = np.zeros(no_runs)
spearman_largest_eig = np.zeros(no_runs)
tau_largest_eig = np.zeros(no_runs)

pearson_corr_values = np.zeros(p**2 * no_runs)
spearman_corr_values = np.zeros(p**2 * no_runs)
tau_corr_values =  np.zeros(p**2 * no_runs)

ks_distance = np.zeros((p, no_runs))

full_correlation_pearson = np.zeros((p, p))
full_correlation_spearman = np.zeros((p, p))
full_correlation_tau = np.zeros((p, p))

full_correlation_pearson_spearman_node_diff = np.zeros((p, no_runs))
full_correlation_pearson_tau_node_diff = np.zeros((p, no_runs))
full_correlation_spearman_tau_node_diff = np.zeros((p, no_runs))

full_correlation_pearson_spearman_edge_diff = np.zeros((no_runs, p, p))
full_correlation_pearson_tau_edge_diff = np.zeros((no_runs, p, p))
full_correlation_spearman_tau_edge_diff = np.zeros((no_runs, p, p))

Ds = []

for x in range(no_runs):
    print("Run %s" % x)

    X_new = X[x*slide_size:x*slide_size+window_size, :]

    if make_normal:
        q = QuantileTransformer(n_quantiles=200, output_distribution='normal')
        X_new = q.fit_transform(X_new)

    Ds.append(np.diag(X_new.std(axis=0)))

    print(X_new.shape)

    pearson_corr = np.corrcoef(X_new.T)
    pearson_largest_eig[x] = scipy.linalg.eigh(pearson_corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    np.fill_diagonal(pearson_corr, 0)
    pearson_corr_values[x*p**2:(x+1)*p**2] = pearson_corr.flatten()
    full_correlation_pearson += pearson_corr

    corrs_pearson.append(pearson_corr)

    spearman_corr, _ = spearmanr(X_new)   
    spearman_largest_eig[x] = scipy.linalg.eigh(spearman_corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    np.fill_diagonal(spearman_corr, 0)
    spearman_corr_values[x*p**2:(x+1)*p**2] = spearman_corr.flatten()
    full_correlation_spearman += spearman_corr
    corrs_spearman.append(spearman_corr)

    tau_corr = calculate_tau_matrix(X_new)   
    tau_largest_eig[x] = scipy.linalg.eigh(tau_corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    tau_corr_values[x*p**2:(x+1)*p**2] = tau_corr.flatten()
    np.fill_diagonal(tau_corr, 0)
    full_correlation_tau += tau_corr
    corrs_tau.append(tau_corr)

    full_correlation_pearson_spearman_node_diff[:, x] = np.abs(pearson_corr/pearson_corr.sum() - spearman_corr/spearman_corr.sum()).sum(axis=0)
    full_correlation_pearson_tau_node_diff[:, x] = np.abs(pearson_corr/pearson_corr.sum() - tau_corr/tau_corr.sum()).sum(axis=0)
    full_correlation_spearman_tau_node_diff[:, x] = np.abs(tau_corr/tau_corr.sum() - spearman_corr/spearman_corr.sum()).sum(axis=0)

    for i in range(p):
        rv = scipy.stats.norm(np.mean(X_new[:, i]), np.std(X_new[:, i]))
        ks_distance[i, x], _ = scipy.stats.kstest(X_new[:, i], rv.cdf)

prev_mst_spearman = None
prev_mst_pearson = None
prev_mst_tau = None

pearson_diffs = np.zeros(no_runs-1)
spearman_diffs = np.zeros(no_runs-1)
tau_diffs = np.zeros(no_runs-1)

maintained_edges_pearson = None
maintained_edges_spearman = None
maintained_edges_tau = None

total_prescence_pearson = np.zeros((p, p))
total_prescence_spearman = np.zeros((p, p))
total_prescence_tau = np.zeros((p, p))

pearson_spearman_diff = np.zeros(no_runs)
pearson_tau_diff = np.zeros(no_runs)
spearman_tau_diff = np.zeros(no_runs)

edges_life_pearson = np.zeros(no_runs)
edges_life_spearman = np.zeros(no_runs)
edges_life_tau = np.zeros(no_runs)

pearson_spearman_node_diff = np.zeros((no_runs, p))
pearson_tau_node_diff = np.zeros((no_runs, p))
spearman_tau_node_diff = np.zeros((no_runs, p))

pearson_spearman_betweenness_node_diff = np.zeros((no_runs, p))
pearson_tau_betweenness_node_diff = np.zeros((no_runs, p))
spearman_tau_betweenness_node_diff = np.zeros((no_runs, p))

first_tree_pearson = None
first_tree_spearman = None
first_tree_tau = None

mean_ol_pearson = np.zeros(no_runs)
mean_ol_spearman = np.zeros(no_runs)
mean_ol_tau = np.zeros(no_runs)

diameter_pearson = np.zeros(no_runs)
diameter_spearman = np.zeros(no_runs)
diameter_tau = np.zeros(no_runs)

average_shortest_path_length_pearson = np.zeros(no_runs)
average_shortest_path_length_spearman = np.zeros(no_runs)
average_shortest_path_length_tau = np.zeros(no_runs)

leaf_fraction_pearson = np.zeros(no_runs)
leaf_fraction_spearman = np.zeros(no_runs)
leaf_fraction_tau = np.zeros(no_runs)

alpha_pearson = np.zeros(no_runs)
alpha_spearman = np.zeros(no_runs)
alpha_tau = np.zeros(no_runs)

sector_degree_centrality_pearson = collections.defaultdict(list)
sector_degree_centrality_spearman = collections.defaultdict(list)
sector_degree_centrality_tau = collections.defaultdict(list)

sector_betweenness_centrality_pearson = collections.defaultdict(list)
sector_betweenness_centrality_spearman = collections.defaultdict(list)
sector_betweenness_centrality_tau = collections.defaultdict(list)

# Just ensure we get the same order of nodes at every graph
nodes = company_names

portfolio_weights_pearson = []
portfolio_weights_spearman = []
portfolio_weights_tau = []
portfolio_weights_full = []

portfolio_risks_pearson = []
portfolio_risks_spearman = []
portfolio_risks_tau = []
portfolio_risks_full = []

portfolio_returns_pearson = []
portfolio_returns_spearman = []
portfolio_returns_tau = []
portfolio_returns_full = []

for i in range(no_runs):
    print("Run %s" % i)
    X_new = X[i*slide_size:i*slide_size+window_size, :]
    X_next = X[(i+1)*slide_size:(i+2)*slide_size+window_size, :]
    cov_test = np.cov(X_next.T)
    mean_test = X_next.mean(axis=0)

    pearson_corr = corrs_pearson[i]
    spearman_corr = corrs_spearman[i]
    tau_corr = corrs_tau[i]
    mst_pearson = topcorr.mst(pearson_corr)
    mst_spearman = topcorr.mst(spearman_corr)
    mst_tau = topcorr.mst(tau_corr)

    mst_pearson_corr = nx.to_numpy_array(mst_pearson)
    mst_spearman_corr = nx.to_numpy_array(mst_spearman)
    mst_tau_corr = nx.to_numpy_array(mst_tau)

    np.fill_diagonal(mst_pearson_corr, 1)
    np.fill_diagonal(mst_spearman_corr, 1)
    np.fill_diagonal(mst_tau_corr, 1)

    alpha = 0.9

    pearson_cov = Ds[i] @ mst_pearson_corr @ Ds[i]
    pearson_cov = alpha * pearson_cov + (1 - alpha) * np.trace(pearson_cov) * np.eye(p) 
    spearman_cov = Ds[i] @ mst_spearman_corr @ Ds[i]  
    spearman_cov = alpha * spearman_cov + (1 - alpha) * np.trace(spearman_cov)  * np.eye(p) 
    tau_cov = Ds[i] @ mst_tau_corr @ Ds[i] 
    tau_cov = alpha * tau_cov  + (1 - alpha) * np.trace(tau_cov) * np.eye(p) 

    _, _, w = calculate_min_risk(pearson_cov, np.zeros(p))
    portfolio_weights_pearson.append(w)
    portfolio_risks_pearson.append(w.T @ cov_test @ w)
    portfolio_returns_pearson.append(w.T @ mean_test)

    _, _, w = calculate_min_risk(spearman_cov, np.zeros(p))
    portfolio_weights_spearman.append(w)
    portfolio_risks_spearman.append(w.T @ cov_test @ w)
    portfolio_returns_spearman.append(w.T @ mean_test)

    _, _, w = calculate_min_risk(tau_cov, np.zeros(p))
    portfolio_weights_tau.append(w)
    portfolio_risks_tau.append(w.T @ cov_test @ w)
    portfolio_returns_tau.append(w.T @ mean_test)

    full_cov = np.cov(X_new.T)
    full_cov = alpha * full_cov + (1 - alpha) * np.trace(full_cov) * np.eye(p)

    _, _, w = calculate_min_risk(full_cov, np.zeros(p))
    portfolio_weights_full.append(w)
    portfolio_risks_full.append(w.T @ cov_test @ w)
    portfolio_returns_full.append(w.T @ mean_test)

    # Build a relabelling dictionary
    node_labels = dict()
    for node in mst_pearson.nodes():
        node_labels[node] = company_names[node]

    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))

    mst_pearson=nx.relabel_nodes(mst_pearson, node_labels)
    nx.set_node_attributes(mst_pearson, node_attributes, 'sector')

    node_labels = dict()
    for node in mst_spearman.nodes():
        node_labels[node] = company_names[node]

    mst_spearman=nx.relabel_nodes(mst_spearman, node_labels)
    nx.set_node_attributes(mst_spearman, node_attributes, 'sector')
    
    node_labels = dict()
    for node in mst_tau.nodes():
        node_labels[node] = company_names[node]

    mst_tau = nx.relabel_nodes(mst_tau, node_labels)
    nx.set_node_attributes(mst_tau, node_attributes, 'sector')

    if i == 0:
        first_tree_pearson = mst_pearson
        first_tree_spearman = mst_spearman
        first_tree_tau = mst_tau
        maintained_edges_pearson = mst_pearson
        maintained_edges_spearman = mst_spearman
        maintained_edges_tau = mst_tau

        nx.write_graphml(mst_pearson, "mst_pearson_%s.graphml" % country)
        nx.write_graphml(mst_spearman, "mst_spearman_%s.graphml" % country)
        nx.write_graphml(mst_tau, "mst_tau_%s.graphml" % country)

    if prev_mst_spearman is not None:
        pearson_diffs[i-1], _ = calculate_mst_diff(mst_pearson, prev_mst_pearson)
        spearman_diffs[i-1], _ = calculate_mst_diff(mst_spearman, prev_mst_spearman)
        tau_diffs[i-1], _ = calculate_mst_diff(mst_tau, prev_mst_tau)

        maintained_edges_pearson = nx.intersection(maintained_edges_pearson, mst_pearson)
        maintained_edges_spearman = nx.intersection(maintained_edges_spearman, mst_spearman)
        maintained_edges_tau = nx.intersection(maintained_edges_tau, mst_tau)

    edges_life_pearson[i] = len(maintained_edges_pearson.edges) / (p-1)
    edges_life_spearman[i] = len(maintained_edges_spearman.edges) / (p-1)
    edges_life_tau[i] = len(maintained_edges_tau.edges) / (p-1)

    pearson_spearman_diff[i], pearson_spearman_node_diff[i] = calculate_mst_diff(mst_pearson, mst_spearman)
    spearman_tau_diff[i], spearman_tau_node_diff[i] = calculate_mst_diff(mst_tau, mst_spearman)
    pearson_tau_diff[i], pearson_tau_node_diff[i] = calculate_mst_diff(mst_tau, mst_pearson)

    M_spearman = nx.to_numpy_array(mst_spearman, nodes)
    M_spearman[np.abs(M_spearman) > 0] = 1
    M_pearson = nx.to_numpy_array(mst_pearson, nodes)
    M_pearson[np.abs(M_pearson) > 0] = 1
    M_tau = nx.to_numpy_array(mst_tau, nodes)
    M_tau[np.abs(M_tau) > 0] = 1
    total_prescence_pearson += M_pearson
    total_prescence_spearman += M_spearman
    total_prescence_tau += M_tau

    prev_mst_spearman = mst_spearman
    prev_mst_pearson = mst_pearson
    prev_mst_tau = mst_tau

    mean_ol_pearson[i] = mean_occupation_layer(mst_pearson)
    mean_ol_spearman[i] = mean_occupation_layer(mst_spearman)
    mean_ol_tau[i] = mean_occupation_layer(mst_tau)

    diameter_pearson[i] = nx.diameter(mst_pearson)
    diameter_spearman[i] = nx.diameter(mst_spearman)
    diameter_tau[i] = nx.diameter(mst_tau)

    average_shortest_path_length_pearson[i] = nx.average_shortest_path_length(correlation_to_distance(mst_pearson), weight='weight')
    average_shortest_path_length_spearman[i] = nx.average_shortest_path_length(correlation_to_distance(mst_spearman), weight='weight')
    average_shortest_path_length_tau[i] = nx.average_shortest_path_length(correlation_to_distance(mst_tau), weight='weight')

    leaf_fraction_pearson[i] = measure_leaf_fraction(mst_pearson)
    leaf_fraction_spearman[i] = measure_leaf_fraction(mst_spearman)
    leaf_fraction_tau[i] = measure_leaf_fraction(mst_tau)

    alpha_pearson[i] = calculate_exponent(mst_pearson)
    alpha_spearman[i] = calculate_exponent(mst_spearman)
    alpha_tau[i] = calculate_exponent(mst_tau)

    # Look at the centralities
    pearson_node, pearson_sector = get_centrality(mst_pearson, degree=True)
    for sec in sector_set:
        sector_degree_centrality_pearson[sec].append(pearson_sector[sec])

    pearson_node = np.array([pearson_node[x] for x in company_names])    

    spearman_node, spearman_sector = get_centrality(mst_spearman, degree=True)
    for sec in sector_set:
        sector_degree_centrality_spearman[sec].append(spearman_sector[sec])

    spearman_node = np.array([spearman_node[x] for x in company_names])     

    tau_node, tau_sector = get_centrality(mst_tau, degree=True)
    for sec in sector_set:
        sector_degree_centrality_tau[sec].append(tau_sector[sec])

    tau_node = np.array([tau_node[x] for x in company_names])     

    pearson_node, pearson_sector = get_centrality(mst_pearson, degree=False)
    
    pearson_node = np.array([pearson_node[x] for x in company_names])     

    for sec in sector_set:
        sector_betweenness_centrality_pearson[sec].append(pearson_sector[sec])

    spearman_node, spearman_sector = get_centrality(mst_spearman, degree=False)
    spearman_node = np.array([spearman_node[x] for x in company_names])     

    for sec in sector_set:
        sector_betweenness_centrality_spearman[sec].append(spearman_sector[sec])

    tau_node, tau_sector = get_centrality(mst_tau, degree=False)
    tau_node = np.array([tau_node[x] for x in company_names])     

    for sec in sector_set:
        sector_betweenness_centrality_tau[sec].append(tau_sector[sec])

plt.figure()
plt.scatter(full_correlation_pearson_spearman_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
#plt.xlim([0, 1])
if country == "DE":
    plt.xlim([0, 0.06])
elif country == "UK":
    plt.xlim([0, 0.02])
else:
    plt.xlim([0, 0.005])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
plt.savefig("pearson_spearman_full_correlation_node_diff_%s.png" % country)
i = spearmanr(full_correlation_pearson_spearman_node_diff.flatten(), ks_distance.flatten())[0]

plt.figure()
plt.scatter(full_correlation_pearson_tau_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
#plt.xlim([0, 1])
if country == "DE":
    plt.xlim([0, 0.06])
elif country == "UK":
    plt.xlim([0, 0.02])
else:
    plt.xlim([0, 0.005])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
plt.savefig("pearson_tau_full_correlation_node_diff_%s.png" % country)
j = spearmanr(full_correlation_pearson_tau_node_diff.flatten(), ks_distance.flatten())[0]

plt.figure()
plt.scatter(full_correlation_spearman_tau_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
if country == "DE":
    plt.xlim([0, 0.06])
elif country == "UK":
    plt.xlim([0, 0.02])
else:
    plt.xlim([0, 0.005])
#plt.xlim([0, 1])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
plt.savefig("spearman_tau_full_correlation_node_diff_%s.png" % country)
k = spearmanr(full_correlation_spearman_tau_node_diff.flatten(), ks_distance.flatten())[0]

print("%.3f & %.3f & %.3f \\\\" % (i, j, k))

plt.figure()
plt.scatter(pearson_spearman_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
plt.xlim([0, 0.2])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
plt.savefig("pearson_spearman_node_diff_%s.png" % country)
i = spearmanr(pearson_spearman_node_diff.flatten(), ks_distance.flatten())[0]

plt.figure()
plt.scatter(pearson_tau_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
plt.xlim([0, 0.2])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
#plt.xlim([0, 1])
plt.savefig("pearson_tau_node_diff_%s.png" % country)
j = spearmanr(pearson_tau_node_diff.flatten(), ks_distance.flatten())[0]

plt.figure()
plt.scatter(spearman_tau_node_diff.flatten(), ks_distance.flatten())
plt.ylim([0, 0.5])
plt.xlim([0, 0.2])
plt.ylabel("KS Distance")
plt.xlabel("Node Difference")
plt.tight_layout()
plt.savefig("spearman_tau_node_diff_%s.png" % country)
k = spearmanr(spearman_tau_node_diff.flatten(), ks_distance.flatten())[0]

print("%.3f & %.3f & %.3f \\\\" % (i, j, k))
print()

plt.figure()
plt.scatter(pearson_corr_values, spearman_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Spearman Correlation Coefficient")
plt.tight_layout()
plt.savefig("pearson_vs_spearman_%s.png" % country)

plt.figure()
plt.scatter(pearson_corr_values, tau_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Kendall $\\tau$ Correlation Coefficient")
plt.tight_layout()
plt.savefig("pearson_vs_tau_%s.png" % country)

plt.figure()
plt.scatter(spearman_corr_values, tau_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Spearman Correlation Coefficient")
plt.ylabel("Kendall $\\tau$ Correlation Coefficient")
plt.tight_layout()
plt.savefig("spearman_vs_tau_%s.png" % country)

max_eig_df = pd.DataFrame()
max_eig_df['Pearson'] = pearson_largest_eig
max_eig_df['Spearman'] = spearman_largest_eig
max_eig_df['$\\tau$'] = tau_largest_eig
max_eig_df.index = dt
max_eig_df.plot()
plt.ylabel("$\lambda_{\max}$")
plt.tight_layout()
plt.savefig("max_eig_%s.png" % country)

edge_life_df = pd.DataFrame()
edge_life_df['Pearson'] = edges_life_pearson
edge_life_df['Spearman'] = edges_life_spearman
edge_life_df['$\\tau$'] = edges_life_tau
edge_life_df.index = dt
edge_life_df.plot()
plt.ylabel("Fraction of Maintained Edges")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("edges_life_%s.png" % country)

mean_occupation_layer_df = pd.DataFrame()
mean_occupation_layer_df['Pearson'] = mean_ol_pearson
mean_occupation_layer_df['Spearman'] = mean_ol_spearman
mean_occupation_layer_df['$\\tau$'] = mean_ol_tau
mean_occupation_layer_df.index = dt
mean_occupation_layer_df.plot()
#plt.title("Mean Occupation Layer")
plt.tight_layout()
plt.savefig("mean_occupation_layer_%s.png" % country)

average_shortest_path_length_df = pd.DataFrame()
average_shortest_path_length_df['Pearson'] = average_shortest_path_length_pearson
average_shortest_path_length_df['Spearman'] = average_shortest_path_length_spearman
average_shortest_path_length_df['$\\tau$'] = average_shortest_path_length_tau

average_shortest_path_length_df.index = dt
average_shortest_path_length_df.plot()
#plt.title("Average Shortest Path Length")
plt.tight_layout()
plt.savefig("average_shortest_path_length_%s.png" % country)

leaf_fraction_df = pd.DataFrame()
leaf_fraction_df['Pearson'] = leaf_fraction_pearson
leaf_fraction_df['Spearman'] = leaf_fraction_spearman
leaf_fraction_df['$\\tau$'] = leaf_fraction_tau
leaf_fraction_df.index = dt
leaf_fraction_df.plot()
#plt.title("Leaf Fraction")
plt.tight_layout()
plt.savefig("leaf_fraction_%s.png" % country)

exponent_df = pd.DataFrame()
exponent_df['Pearson'] = alpha_pearson
exponent_df['Spearman'] = alpha_spearman
exponent_df['$\\tau$'] = alpha_tau
exponent_df.index = dt
exponent_df.plot()
#plt.title("Exponent")
plt.tight_layout()
plt.savefig("exponent_%s.png" % country)

plt.figure()
diff_df = pd.DataFrame()
diff_df['Pearson - Spearman'] = pd.Series(pearson_spearman_diff)
diff_df['Pearson - $\\tau$'] = pd.Series(pearson_tau_diff)
diff_df['Spearman - $\\tau$'] = pd.Series(spearman_tau_diff)
diff_df.index = dt
diff_df.plot()
plt.ylim([0.0, 0.6])
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("edge_difference_%s.png" % country)

edge_life_df = pd.DataFrame()
edge_life_df['Pearson'] = edges_life_pearson
edge_life_df['Spearman'] = edges_life_spearman
edge_life_df['$\\tau$'] = edges_life_tau
edge_life_df.index = dt
edge_life_df.plot()
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("edges_life_%s.png" % country)

diff_df = pd.DataFrame()
diff_df['Pearson'] = pearson_diffs
diff_df['Spearman'] = spearman_diffs
diff_df['$\\tau$'] = tau_diffs
diff_df.index = dt_2
diff_df.plot()
plt.ylim([0.0, 0.5])
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("diff_%s.png" % country)

# Look at the edges that are maintained throughout the dataset
print("Total maintained Pearson")
print(np.count_nonzero(np.triu(total_prescence_pearson) == no_runs))
print("Total maintained Spearman")
print(np.count_nonzero(np.triu(total_prescence_spearman) == no_runs))
print("Total maintained Tau")
print(np.count_nonzero(np.triu(total_prescence_tau) == no_runs))
# Which edges are these?
print("Pearson")
ind = np.where(np.triu(total_prescence_pearson) == no_runs)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 

print("Spearman")
ind = np.where(np.triu(total_prescence_spearman) == no_runs)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 

print("Tau")
ind = np.where(np.triu(total_prescence_tau) == no_runs)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 

sector_degree_centrality_pearson_df = pd.DataFrame.from_dict(sector_degree_centrality_pearson)
sector_degree_centrality_pearson_df.index = dt
sector_degree_centrality_spearman_df = pd.DataFrame.from_dict(sector_degree_centrality_spearman)
sector_degree_centrality_spearman_df.index = dt
sector_degree_centrality_tau_df = pd.DataFrame.from_dict(sector_degree_centrality_tau)
sector_degree_centrality_tau_df.index = dt

sector_betweenness_centrality_pearson_df = pd.DataFrame.from_dict(sector_betweenness_centrality_pearson)
sector_betweenness_centrality_pearson_df.index = dt
sector_betweenness_centrality_spearman_df = pd.DataFrame.from_dict(sector_betweenness_centrality_spearman)
sector_betweenness_centrality_spearman_df.index = dt
sector_betweenness_centrality_tau_df = pd.DataFrame.from_dict(sector_betweenness_centrality_tau)
sector_betweenness_centrality_tau_df.index = dt

max_val = max(sector_degree_centrality_pearson_df.max().max(), sector_degree_centrality_spearman_df.max().max(), sector_degree_centrality_tau_df.max().max())

text_size = 22
colour_scheme = 'YlOrRd'
# We use plotly to create some heatmaps that look a bit better than just straight graphs
import plotly.graph_objects as go
import datetime
dates = sector_degree_centrality_pearson_df.index    
sectors = sector_degree_centrality_pearson_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_pearson_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("pearson_degree_centrality_%s.png" % country)

dates = sector_degree_centrality_spearman_df.index    
sectors = sector_degree_centrality_spearman_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_spearman_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("spearman_degree_centrality_%s.png" % country)
dates = sector_degree_centrality_tau_df.index    
sectors = sector_degree_centrality_tau_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_tau_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("tau_degree_centrality_%s.png" % country)

max_val = max(sector_betweenness_centrality_pearson_df.max().max(), sector_betweenness_centrality_spearman_df.max().max(), sector_betweenness_centrality_tau_df.max().max())

dates = sector_betweenness_centrality_pearson_df.index    
sectors = sector_betweenness_centrality_pearson_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_pearson_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("pearson_betweenness_centrality_%s.png" % country)

dates = sector_betweenness_centrality_spearman_df.index    
sectors = sector_betweenness_centrality_spearman_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_spearman_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("spearman_betweenness_centrality_%s.png" % country)

dates = sector_betweenness_centrality_tau_df.index    
sectors = sector_betweenness_centrality_tau_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_tau_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale=colour_scheme)) 
fig.update_layout(font=dict(size=text_size))
fig.write_image("tau_betweenness_centrality_%s.png" % country)

portfolio_diff_pearson = np.zeros(no_runs)
portfolio_diff_spearman = np.zeros(no_runs)
portfolio_diff_tau = np.zeros(no_runs)
portfolio_diff_full = np.zeros(no_runs)

for i in range(no_runs-1):
    portfolio_diff_pearson[i] = np.linalg.norm(portfolio_weights_pearson[i+1] - portfolio_weights_pearson[i], 1)
    portfolio_diff_spearman[i] = np.linalg.norm(portfolio_weights_spearman[i+1] - portfolio_weights_spearman[i], 1)
    portfolio_diff_tau[i] = np.linalg.norm(portfolio_weights_tau[i+1] - portfolio_weights_tau[i], 1)
    portfolio_diff_full[i] = np.linalg.norm(portfolio_weights_full[i+1] - portfolio_weights_full[i], 1)

portfolio_returns = pd.DataFrame()
portfolio_returns['Pearson'] = portfolio_returns_pearson
portfolio_returns['Spearman'] = portfolio_returns_spearman
portfolio_returns['$\\tau$'] = portfolio_returns_tau
portfolio_returns['Full'] = portfolio_returns_full
portfolio_returns.index = dt
portfolio_returns.plot()
plt.ylabel("Returns")
plt.tight_layout()
plt.savefig("portfolio_returns_%s.png" % country)

portfolio_risks = pd.DataFrame()
portfolio_risks['Pearson'] = portfolio_risks_pearson
portfolio_risks['Spearman'] = portfolio_risks_spearman
portfolio_risks['$\\tau$'] = portfolio_risks_tau
portfolio_risks['Full'] = portfolio_risks_full
portfolio_risks.index = dt
portfolio_risks.plot()
plt.ylabel("Risks")
plt.tight_layout()
plt.savefig("portfolio_risks_%s.png" % country)

portfolio_sharpe_pearson = np.divide(portfolio_returns_pearson, portfolio_risks_pearson)
portfolio_sharpe_spearman = np.divide(portfolio_returns_spearman, portfolio_risks_spearman)
portfolio_sharpe_tau = np.divide(portfolio_returns_tau, portfolio_risks_tau)
portfolio_sharpe_full = np.divide(portfolio_returns_full, portfolio_risks_full)

portfolio_risks = pd.DataFrame()
portfolio_risks['Pearson'] = portfolio_sharpe_pearson
portfolio_risks['Spearman'] = portfolio_sharpe_spearman
portfolio_risks['$\\tau$'] = portfolio_sharpe_tau
portfolio_risks['Full'] = portfolio_sharpe_full
portfolio_risks.index = dt
portfolio_risks.plot()
plt.ylabel("Sharpe Ratio")
plt.tight_layout()
plt.savefig("portfolio_sharpe_%s.png" % country)

portfolio_risks = pd.DataFrame()
portfolio_risks['Pearson'] = portfolio_diff_pearson
portfolio_risks['Spearman'] = portfolio_diff_spearman
portfolio_risks['$\\tau$'] = portfolio_diff_tau
portfolio_risks['Full'] = portfolio_diff_full
portfolio_risks.index = dt
portfolio_risks.plot()
plt.ylabel("Turnover")
plt.tight_layout()
plt.savefig("portfolio_turnover_%s.png" % country)

print("Mean Portfolio Diff")
print(portfolio_diff_pearson.mean())
print(portfolio_diff_spearman.mean())
print(portfolio_diff_tau.mean())
print(portfolio_diff_full.mean())

print("Mean Sharpe Ratio")
print(portfolio_sharpe_pearson.mean())
print(portfolio_sharpe_spearman.mean())
print(portfolio_sharpe_tau.mean())
print(portfolio_sharpe_full.mean())

plt.close('all')