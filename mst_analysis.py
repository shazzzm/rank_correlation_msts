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

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

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

def get_sector_full_nice_name(sector):
    """
    Returns a short version of the sector name
    """       
    if sector == "information_technology":
        return "Information Technology"
    elif sector == "real_estate":
        return "Real Estate"
    elif sector == "materials":
        return "Materials"
    elif sector == "telecommunication_services":
        return "Telecommunication Services"
    elif sector == "energy":
        return "Energy"
    elif sector == "financials":
        return "Financials"
    elif sector == "utilities":
        return "Utilities"
    elif sector == "industrials":
        return "Industrials"
    elif sector == "consumer_discretionary":
        return "Consumer Discretionary"
    elif sector == "health_care":
        return "Healthcare"
    elif sector == "consumer_staples":
        return "Consumer Staples"
    else:
        raise Exception("%s is not a valid sector" % sector)

def calculate_mst_diff(mst, prev_mst):
    nodes = list(mst.nodes)
    M = nx.to_numpy_array(mst, nodes)
    p = M.shape[0]
    M_prev = nx.to_numpy_array(prev_mst, nodes)

    M[np.abs(M) > 0] = 1
    M_prev[np.abs(M_prev) > 0] = 1

    fraction_total_diff = np.count_nonzero(M - M_prev)/(4*(p-1))

    M_diff = M - M_prev
    node_diff = np.abs(M_diff.sum(axis=0))
    fraction_node_diff = np.divide(node_diff, M.sum(axis=0))
    return fraction_total_diff, node_diff, fraction_node_diff

def correlation_to_distance(G):
    """
    Converts a correlation graph to a distance based one
    """
    G = G.copy()
    for edge in G.edges():
        G.edges[edge]['weight'] =  np.sqrt(2 - 2*G.edges[edge]['weight'])
    return G

def measure_node_difference(mst_1, mst_2):
    nodes = list(mst_1.nodes)
    M_1 = nx.to_numpy_array(mst_1, nodes)
    M_2 = nx.to_numpy_array(mst_2, nodes)

    p = M_1.shape[0]

    M_1[np.abs(M_1) > 0] = 1
    M_2[np.abs(M_2) > 0] = 1

    node_difference = (M_1 - M_2).sum(axis=0)
    node_degree_one = M_1.sum(axis=0)
    node_degree_two = M_2.sum(axis=0)
    
    degree_diff_corr, _ = spearmanr(node_degree_one, node_difference)
    degree_corr, _ = spearmanr(node_degree_one, node_degree_two)

    # Threshold the adjacency matrix so we only get high degree nodes
    node_degree_one[node_degree_one > 4] = 0
    node_degree_two[node_degree_two > 4] = 0
    threshold_degree_corr, _ = spearmanr(node_degree_one, node_degree_two)

    return degree_diff_corr, degree_corr, threshold_degree_corr

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

    return node_centrality, sector_centrality

# Set the country you desire to analyze
country = "US"
df = pd.read_csv("S&P500.csv", index_col=0)
window_size = 252*2

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values

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


slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])
dt = pd.to_datetime(dates)
dt_2 = dt[1:]

Graphs_spearman = []
Graphs_pearson = []
Graphs_tau = []

pearson_largest_eig = np.zeros(no_runs)
spearman_largest_eig = np.zeros(no_runs)
tau_largest_eig = np.zeros(no_runs)

pearson_corr_values = np.zeros(p**2 * no_runs)
spearman_corr_values = np.zeros(p**2 * no_runs)
tau_corr_values =  np.zeros(p**2 * no_runs)

stdev = np.zeros((p, no_runs))
kurtosis = np.zeros((p, no_runs))

for x in range(no_runs):
    print("Run %s" % x)
    X_new = X[x*slide_size:(x+1)*slide_size+window_size, :]
    stdev[:, x] = np.std(X_new, axis=0)
    kurtosis[:, x] = scipy.stats.kurtosis(X_new, axis=0)
    corr = np.corrcoef(X_new.T)
    pearson_largest_eig[x] = scipy.linalg.eigh(corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    np.fill_diagonal(corr, 0)
    pearson_corr_values[x*p**2:(x+1)*p**2] = corr.flatten()

    G=nx.from_numpy_matrix(corr)
    G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G, node_attributes, 'sector')
    Graphs_pearson.append(G)

    corr, _ = spearmanr(X_new)   
    spearman_largest_eig[x] = scipy.linalg.eigh(corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    np.fill_diagonal(corr, 0)
    spearman_corr_values[x*p**2:(x+1)*p**2] = corr.flatten()


    G=nx.from_numpy_matrix(corr)
    G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G, node_attributes, 'sector')
    Graphs_spearman.append(G)

    corr = calculate_tau_matrix(X_new)   
    tau_largest_eig[x] = scipy.linalg.eigh(corr, eigvals_only=True, eigvals=(p-1, p-1))[0]
    tau_corr_values[x*p**2:(x+1)*p**2] = corr.flatten()
    np.fill_diagonal(corr, 0)

    G=nx.from_numpy_matrix(corr)
    G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G, node_attributes, 'sector')
    Graphs_tau.append(G)

prev_mst_spearman = None
prev_mst_pearson = None
prev_mst_tau = None

pearson_diffs = np.zeros(no_runs-1)
spearman_diffs = np.zeros(no_runs-1)
tau_diffs = np.zeros(no_runs-1)

total_prescence_pearson = np.zeros((p, p))
total_prescence_spearman = np.zeros((p, p))
total_prescence_tau = np.zeros((p, p))

pearson_spearman_diff = np.zeros(no_runs)
pearson_tau_diff = np.zeros(no_runs)
spearman_tau_diff = np.zeros(no_runs)

degree_diff_corr = np.zeros(no_runs)
pearson_spearman_degree_corr = np.zeros(no_runs)
pearson_tau_degree_corr = np.zeros(no_runs)
spearman_tau_degree_corr = np.zeros(no_runs)

pearson_spearman_threshold_degree_corr = np.zeros(no_runs)
pearson_tau_threshold_degree_corr = np.zeros(no_runs)
spearman_tau_threshold_degree_corr = np.zeros(no_runs)

edges_life_pearson = np.zeros(no_runs)
edges_life_spearman = np.zeros(no_runs)
edges_life_tau = np.zeros(no_runs)

pearson_spearman_node_diff = np.zeros((no_runs, p))
pearson_tau_node_diff = np.zeros((no_runs, p))
spearman_tau_node_diff = np.zeros((no_runs, p))

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

betweeness_centrality_agreement_pearson_spearman = np.zeros(no_runs)
betweeness_centrality_agreement_pearson_tau = np.zeros(no_runs)
betweeness_centrality_agreement_spearman_tau = np.zeros(no_runs)

# Just ensure we get the same order of nodes at every graph
nodes = list(Graphs_pearson[0].nodes)

for i, (G_pearson, G_spearman, G_tau) in enumerate(zip(Graphs_pearson, Graphs_spearman, Graphs_tau)):
    mst_pearson = nx.minimum_spanning_tree(correlation_to_distance(G_pearson))
    mst_spearman = nx.minimum_spanning_tree(correlation_to_distance(G_spearman))
    mst_tau = nx.minimum_spanning_tree(correlation_to_distance(G_tau))
    if i == 0:
        first_tree_pearson = mst_pearson
        first_tree_spearman = mst_spearman
        first_tree_tau = mst_tau

    if prev_mst_spearman is not None:
        pearson_diffs[i-1], _, _ = calculate_mst_diff(mst_pearson, prev_mst_pearson)
        spearman_diffs[i-1], _, _ = calculate_mst_diff(mst_spearman, prev_mst_spearman)
        tau_diffs[i-1], _, _ = calculate_mst_diff(mst_tau, prev_mst_tau)

    pearson_spearman_diff[i], pearson_spearman_node_diff[i], _ = calculate_mst_diff(mst_pearson, mst_spearman)
    spearman_tau_diff[i], pearson_tau_node_diff[i], _ = calculate_mst_diff(mst_tau, mst_spearman)
    pearson_tau_diff[i], spearman_tau_node_diff[i], _ = calculate_mst_diff(mst_tau, mst_pearson)

    degree_diff_corr[i], pearson_spearman_degree_corr[i], pearson_spearman_threshold_degree_corr[i] = measure_node_difference(mst_pearson, mst_spearman)
    degree_diff_corr[i], pearson_tau_degree_corr[i], pearson_tau_threshold_degree_corr[i] = measure_node_difference(mst_pearson, mst_tau)
    degree_diff_corr[i], spearman_tau_degree_corr[i], spearman_tau_threshold_degree_corr[i] = measure_node_difference(mst_spearman, mst_tau)

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

    edges_life_pearson[i], _, _ = calculate_mst_diff(mst_pearson, first_tree_pearson)
    edges_life_spearman[i], _, _ = calculate_mst_diff(mst_spearman, first_tree_spearman)
    edges_life_tau[i], _, _ = calculate_mst_diff(mst_spearman, first_tree_tau)

    mean_ol_pearson[i] = mean_occupation_layer(mst_pearson)
    mean_ol_spearman[i] = mean_occupation_layer(mst_spearman)
    mean_ol_tau[i] = mean_occupation_layer(mst_tau)

    diameter_pearson[i] = nx.diameter(mst_pearson)
    diameter_spearman[i] = nx.diameter(mst_spearman)
    diameter_tau[i] = nx.diameter(mst_tau)

    average_shortest_path_length_pearson[i] = nx.average_shortest_path_length(mst_pearson, weight='weight')
    average_shortest_path_length_spearman[i] = nx.average_shortest_path_length(mst_spearman, weight='weight')
    average_shortest_path_length_tau[i] = nx.average_shortest_path_length(mst_tau, weight='weight')

    leaf_fraction_pearson[i] = measure_leaf_fraction(mst_pearson)
    leaf_fraction_spearman[i] = measure_leaf_fraction(mst_spearman)
    leaf_fraction_tau[i] = measure_leaf_fraction(mst_tau)

    alpha_pearson[i] = calculate_exponent(mst_pearson)
    alpha_spearman[i] = calculate_exponent(mst_spearman)
    alpha_tau[i] = calculate_exponent(mst_tau)

    # Look at the centralities
    pearson_node, pearson_sector = get_centrality(mst_pearson, degree=True)
    for sec in pearson_sector:
        sector_degree_centrality_pearson[sec].append(pearson_sector[sec])

    spearman_node, spearman_sector = get_centrality(mst_spearman, degree=True)
    for sec in spearman_sector:
        sector_degree_centrality_spearman[sec].append(spearman_sector[sec])

    tau_node, tau_sector = get_centrality(mst_tau, degree=True)
    for sec in tau_sector:
        sector_degree_centrality_tau[sec].append(tau_sector[sec])

    pearson_node, pearson_sector = get_centrality(mst_pearson, degree=False)
    for sec in pearson_sector:
        sector_betweenness_centrality_pearson[sec].append(pearson_sector[sec])

    spearman_node, spearman_sector = get_centrality(mst_spearman, degree=False)
    for sec in spearman_sector:
        sector_betweenness_centrality_spearman[sec].append(spearman_sector[sec])

    tau_node, tau_sector = get_centrality(mst_tau, degree=False)
    for sec in tau_sector:
        sector_betweenness_centrality_tau[sec].append(tau_sector[sec])

plt.figure()
plt.scatter(pearson_corr_values, spearman_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Spearman Correlation Coefficient")
plt.tight_layout()
plt.savefig("pearson_vs_spearman.png")

plt.figure()
plt.scatter(pearson_corr_values, tau_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Kendall $\\tau$ Correlation Coefficient")
plt.tight_layout()
plt.savefig("pearson_vs_tau.png")

plt.figure()
plt.scatter(spearman_corr_values, tau_corr_values)
plt.xlim([-0.3, 1])
plt.ylim([-0.3, 1])
plt.xlabel("Spearman Correlation Coefficient")
plt.ylabel("Kendall $\\tau$ Correlation Coefficient")
plt.tight_layout()
plt.savefig("spearman_vs_tau.png")

# Create some example MSTs for us to draw
G_pearson = Graphs_pearson[0]
G_spearman = Graphs_spearman[0]
G_tau = Graphs_tau[0]

mst_pearson = nx.minimum_spanning_tree(correlation_to_distance(G_pearson))
mst_spearman = nx.minimum_spanning_tree(correlation_to_distance(G_spearman))
mst_tau = nx.minimum_spanning_tree(correlation_to_distance(G_tau))

nx.write_graphml(mst_pearson, "mst_pearson_0.graphml")
nx.write_graphml(mst_spearman, "mst_spearman_0.graphml")
nx.write_graphml(mst_tau, "mst_tau_0.graphml")

max_eig_df = pd.DataFrame()
max_eig_df['Pearson'] = pearson_largest_eig
max_eig_df['Spearman'] = spearman_largest_eig
max_eig_df['$\\tau$'] = tau_largest_eig
max_eig_df.index = dt
max_eig_df.plot()
plt.ylabel("$\lambda_{\max}$")
plt.tight_layout()
plt.savefig("max_eig.png")

edge_life_df = pd.DataFrame()
edge_life_df['Pearson'] = edges_life_pearson
edge_life_df['Spearman'] = edges_life_spearman
edge_life_df['$\\tau$'] = edges_life_tau
edge_life_df.index = dt
edge_life_df.plot()
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("edges_life.png")

mean_occupation_layer_df = pd.DataFrame()
mean_occupation_layer_df['Pearson'] = mean_ol_pearson
mean_occupation_layer_df['Spearman'] = mean_ol_spearman
mean_occupation_layer_df['$\\tau$'] = mean_ol_tau
mean_occupation_layer_df.index = dt
mean_occupation_layer_df.plot()
#plt.title("Mean Occupation Layer")
plt.tight_layout()
plt.savefig("mean_occupation_layer.png")

average_shortest_path_length_df = pd.DataFrame()
average_shortest_path_length_df['Pearson'] = average_shortest_path_length_pearson
average_shortest_path_length_df['Spearman'] = average_shortest_path_length_spearman
average_shortest_path_length_df['$\\tau$'] = average_shortest_path_length_tau

average_shortest_path_length_df.index = dt
average_shortest_path_length_df.plot()
#plt.title("Average Shortest Path Length")
plt.tight_layout()
plt.savefig("average_shortest_path_length.png")

leaf_fraction_df = pd.DataFrame()
leaf_fraction_df['Pearson'] = leaf_fraction_pearson
leaf_fraction_df['Spearman'] = leaf_fraction_spearman
leaf_fraction_df['$\\tau$'] = leaf_fraction_tau
leaf_fraction_df.index = dt
leaf_fraction_df.plot()
#plt.title("Leaf Fraction")
plt.tight_layout()
plt.savefig("leaf_fraction.png")

exponent_df = pd.DataFrame()
exponent_df['Pearson'] = alpha_pearson
exponent_df['Spearman'] = alpha_spearman
exponent_df['$\\tau$'] = alpha_tau
exponent_df.index = dt
exponent_df.plot()
#plt.title("Exponent")
plt.tight_layout()
plt.savefig("exponent.png")

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
plt.savefig("edge_difference.png")

edge_life_df = pd.DataFrame()
edge_life_df['Pearson'] = edges_life_pearson
edge_life_df['Spearman'] = edges_life_spearman
edge_life_df['$\\tau$'] = edges_life_tau
edge_life_df.index = dt
edge_life_df.plot()
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("edges_life.png")

diff_df = pd.DataFrame()
diff_df['Pearson'] = pearson_diffs
diff_df['Spearman'] = spearman_diffs
diff_df['$\\tau$'] = tau_diffs
diff_df.index = dt_2
diff_df.plot()
#plt.title("Edge Difference")
plt.ylabel("Fraction of Different Edges")
plt.tight_layout()
plt.savefig("diff.png")

plt.figure()
degree_corr_df = pd.DataFrame()
degree_corr_df['Pearson - Spearman'] = pd.Series(pearson_spearman_degree_corr)
degree_corr_df['Pearson - $\\tau$'] = pd.Series(pearson_tau_degree_corr)
degree_corr_df['Spearman - $\\tau$'] = pd.Series(spearman_tau_degree_corr)
degree_corr_df.index = dt
degree_corr_df.plot()
plt.ylim([0.2, 1])
#plt.title("Correlation between node degree")
plt.ylabel("Degree Correlation")
plt.tight_layout()
plt.savefig("degree_corr.png")

plt.figure()
degree_corr_df = pd.DataFrame()
degree_corr_df['Pearson - Spearman'] = pd.Series(pearson_spearman_threshold_degree_corr)
degree_corr_df['Pearson - $\\tau$'] = pd.Series(pearson_tau_threshold_degree_corr)
degree_corr_df['Spearman - $\\tau$'] = pd.Series(spearman_tau_threshold_degree_corr)
degree_corr_df.index = dt
degree_corr_df.plot()
plt.ylim([0.2, 1])
plt.ylabel("Degree Correlation")
plt.tight_layout()
#plt.title("Correlation between node degree")
plt.savefig("peripheries_degree_corr.png")

# Look at the edges that are maintained throughout the dataset
print("Total maintained Pearson")
print(np.count_nonzero(np.triu(total_prescence_pearson) == 142))
print("Total maintained Spearman")
print(np.count_nonzero(np.triu(total_prescence_spearman) == 142))
print("Total maintained Tau")
print(np.count_nonzero(np.triu(total_prescence_tau) == 142))
# Which edges are these?
print("Pearson")
ind = np.where(np.triu(total_prescence_pearson) == 142)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 

print("Spearman")
ind = np.where(np.triu(total_prescence_spearman) == 142)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 

print("Tau")
ind = np.where(np.triu(total_prescence_tau) == 142)
xs = ind[0]
ys = ind[1]

for x, y in zip(xs, ys):
    print("%s - %s (%s - %s)" % (company_names[x], company_names[y], company_sectors[x], company_sectors[y])) 


# Which nodes have the highest degree overall?
total_degree_spearman = total_prescence_spearman.sum(axis=0)
total_degree_pearson = total_prescence_pearson.sum(axis=0)
total_degree_tau = total_prescence_tau.sum(axis=0)

spearman_ind = np.argsort(total_degree_spearman)[::-1]
pearson_ind =  np.argsort(total_degree_pearson)[::-1]
tau_ind =  np.argsort(total_degree_tau)[::-1]

print("Pearson most central")
for i in pearson_ind[0:10]:
    print("%s & %s & %s \\\\" % (company_names[i], company_sectors[i], total_degree_pearson[i]) )
    if i == 10:
        break

print()
print("Spearman most central")
for i in spearman_ind[0:10]:
    print("%s & %s & %s\\\\" % (company_names[i], company_sectors[i], total_degree_spearman[i]) )

print()
print("Tau most central")
for i in tau_ind[0:10]:
    print("%s & %s & %s\\\\" % (company_names[i], company_sectors[i], total_degree_tau[i]) )

print("Total degree correlation")
print("Pearson - Spearman")
print(spearmanr(total_degree_pearson, total_degree_spearman))
print("Pearson - tau")
print(spearmanr(total_degree_pearson, total_degree_tau))
print("tau - Spearman")
print(spearmanr(total_degree_tau, total_degree_spearman))

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

# We use plotly to create some heatmaps that look a bit better than just straight graphs
import plotly.graph_objects as go
import datetime
dates = sector_degree_centrality_pearson_df.index    
sectors = sector_degree_centrality_pearson_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_pearson_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

dates = sector_degree_centrality_spearman_df.index    
sectors = sector_degree_centrality_spearman_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_spearman_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

dates = sector_degree_centrality_tau_df.index    
sectors = sector_degree_centrality_tau_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_degree_centrality_tau_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

max_val = max(sector_betweenness_centrality_pearson_df.max().max(), sector_betweenness_centrality_spearman_df.max().max(), sector_betweenness_centrality_tau_df.max().max())

dates = sector_betweenness_centrality_pearson_df.index    
sectors = sector_betweenness_centrality_pearson_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_pearson_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

dates = sector_betweenness_centrality_spearman_df.index    
sectors = sector_betweenness_centrality_spearman_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_spearman_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

dates = sector_betweenness_centrality_tau_df.index    
sectors = sector_betweenness_centrality_tau_df.keys().values
fig = go.Figure(data=go.Heatmap(z=sector_betweenness_centrality_tau_df.T.values, x= dates.to_pydatetime(), zmax=max_val, zmin=0, y=sectors, colorscale='Viridis')) 
fig.show()

pearson_spearman = pd.Series()
pearson_tau = pd.Series()
spearman_tau = pd.Series()
# Measure the correlation between them all
print("Degree Centrality Correlation")
for sec in sector_degree_centrality_tau_df:
    pearson_spearman[sec] = spearmanr(sector_degree_centrality_pearson_df[sec], sector_degree_centrality_spearman[sec])[0]
    pearson_tau[sec] = spearmanr(sector_degree_centrality_pearson_df[sec], sector_degree_centrality_tau_df[sec])[0]
    spearman_tau[sec] = spearmanr(sector_degree_centrality_tau_df[sec], sector_degree_centrality_spearman[sec])[0]
    print("%s & %.3f & %.3f & %.3f \\\\" % (sec, pearson_spearman[sec], pearson_tau[sec], spearman_tau[sec]))

print("Pearson - Spearman $%.3f \pm %.3f$" % (pearson_spearman.mean(), pearson_spearman.std()))
print("Pearson - $\\tau %.3f \pm %.3f$" % (pearson_tau.mean(), pearson_tau.std()))
print("Spearman - $\\tau $%.3f \pm %.3f$" % (spearman_tau.mean(), spearman_tau.std()))


pearson_spearman = pd.Series()
pearson_tau = pd.Series()
spearman_tau = pd.Series()
# Measure the correlation between them all
print("Betweenness Centrality Correlation")
for sec in sector_betweenness_centrality_pearson_df:
    pearson_spearman[sec] = spearmanr(sector_betweenness_centrality_pearson_df[sec], sector_betweenness_centrality_spearman[sec])[0]
    pearson_tau[sec] = spearmanr(sector_betweenness_centrality_pearson_df[sec], sector_betweenness_centrality_tau[sec])[0]
    spearman_tau[sec] = spearmanr(sector_betweenness_centrality_tau_df[sec], sector_betweenness_centrality_spearman[sec])[0]
    print("%s & %.3f & %.3f & %.3f \\\\" % (sec, pearson_spearman[sec], pearson_tau[sec], spearman_tau[sec]))

print("Pearson - Spearman $%.3f \pm %.3f$" % (pearson_spearman.mean(), pearson_spearman.std()))
print("Pearson - $\\tau %.3f \pm %.3f$" % (pearson_tau.mean(), pearson_tau.std()))
print("Spearman - $\\tau $%.3f \pm %.3f$" % (spearman_tau.mean(), spearman_tau.std()))

plt.close('all')