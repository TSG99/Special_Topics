#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

#Parameters
DATA_FILE       = r"C:\Users\shein\Downloads\SWOW-EN18\strength.SWOW-EN.R123.20180827.csv"
NUM_WALKERS     = 141
STEPS_PER_WALK  = 2000
RESTART_PROB    = 0.05
SIZES_TO_TEST   = [10, 25, 50, 100, 166]  

#Animal Set
full_animals = sorted({
    'aardvark','abacus','abalone','admiral','african_elephant','albatross','alligator','alpaca','anaconda',
    'angelfish','ant','anteater','antelope','armadillo','axolotl','baboon','barracuda','basset_hound','bat',
    'beagle','bear','beetle','beluga_whale','bison','black_panther','black_widow_spider','blowfish','bobcat','buffalo',
    'bulldog','bullfrog','butterfly','coyote','chicken','chinchilla','chimpanzee','cockroach','cod','cougar','cow',
    'crab','crocodile','crow','deer','dingo','dodo','dog','dolphin','donkey','dove','duck','eagle','eel',
    'elephant','elk','emu','falcon','ferret','finch','flamingo','fly','fox','frog','gazelle','geese','gerbil',
    'giraffe','goat','goldfish','goose','gorilla','grasshopper','guinea_pig','hamster','hare','hawk','hippopotamus',
    'hornet','horse','hummingbird','hyena','ibex','iguana','impala','jaguar','jellyfish','kangaroo','koala','komodo_dragon',
    'kookaburra','ladybug','lamb','lark','leopard','lion','lizard','llama','lobster','locust','mongoose','monkey',
    'moose','moth','mountain_lion','mouse','octopus','orangutan','ostrich','otter','owl','ox','panda','parrot','peacock',
    'penguin','pig','pigeon','polar_bear','porcupine','quail','rabbit','raccoon','rat','raven','reindeer','rhinoceros',
    'roadrunner','rooster','salmon','scorpion','seal','shark','sheep','shrimp','skunk','slug','snail','snake','sparrow',
    'spider','squid','squirrel','starling','stingray','stoat','stork','swan','termite','toad','tortoise','toucan','tree_frog',
    'trout','tuna','turkey','turtle','vulture','wallaby','walrus','wasp','weasel','whale','wolf','wolverine','woodpecker',
    'zebra'
})

#Build Graph
df = pd.read_csv(DATA_FILE, sep='\t')
G_full = nx.Graph()
for _, row in df.iterrows():
    cue, resp = row['cue'], row['response']
    if isinstance(cue, str) and isinstance(resp, str):
        cue, resp = cue.lower(), resp.lower()
        G_full.add_edge(cue, resp, weight=row['N'])

print(f"Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

#Walk
def random_walk_limited_animals(G, known_animals, start, steps, restart_prob):
    seen = set()
    first_hits = {}
    cur = start
    for t in range(steps):
        if cur in known_animals and cur not in seen:
            seen.add(cur)
            first_hits[cur] = t
        if np.random.rand() < restart_prob or cur not in G:
            cur = start
            continue
        nbrs = list(G[cur])
        if not nbrs:
            cur = start
            continue
        wts = np.array([G[cur][n].get('weight', 1.0) for n in nbrs], float)
        probs = wts / wts.sum()
        cur = np.random.choice(nbrs, p=probs)
    return len(seen)


results = []
for size in SIZES_TO_TEST:
    subset = set(random.sample(full_animals, size))
    retrieved = []
    for _ in range(NUM_WALKERS):
        count = random_walk_limited_animals(
            G_full, subset, start='animal',
            steps=STEPS_PER_WALK, restart_prob=RESTART_PROB
        )
        retrieved.append(count)
    avg = np.mean(retrieved)
    results.append((size, avg))
    print(f"Known: {size} → Retrieved: {avg:.2f} animals")


x, y = zip(*results)
plt.figure(figsize=(7, 5))
plt.plot(x, y, marker='o', label='Simulation')
plt.title("Effect of Knowledge Size on Animal Retrieval")
plt.xlabel("Number of Known Animals")
plt.ylabel("Mean Retrieved Animals")
plt.grid(True)
plt.tight_layout()
plt.show()


known_arr = np.array(x)
retrieved_arr = np.array(y)
log_k = np.log(known_arr)
log_r = np.log(retrieved_arr)


b, log_a = np.polyfit(log_k, log_r, 1)
a = np.exp(log_a)

print(f"\nPower-law form: Retrieved ≈ {a:.3f} * Known^{b:.3f}")
print(f"Log–log form: ln(Retrieved) = {b:.3f} * ln(Known) + {log_a:.3f}")

known_range = np.linspace(known_arr.min(), known_arr.max(), 100)
fitted = a * known_range**b

plt.figure(figsize=(7, 5))
plt.scatter(known_arr, retrieved_arr, color='blue', label='Data')
plt.plot(known_range, fitted, color='red',
         label=f'Fit: y={a:.3f}·x^{b:.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Known Animals (log scale)")
plt.ylabel("Mean Retrieved Animals (log scale)")
plt.title("Log-Log Power Law Fit")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# In[22]:


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from networkx.algorithms.community import greedy_modularity_communities
import collections

#Parameters
DATA_FILE        = r"C:\Users\shein\Downloads\SWOW-EN18\strength.SWOW-EN.R123.20180827.csv"
NUM_WALKERS      = 141
STEPS_PER_WALK   = 2000
RESTART_PROB     = 0.05
IRT_THRESHOLD    = 5.0
TEST_CLUSTER_SIZES = [3, 5, 7, 10, 15] 

#Animal Set
full_animals = sorted({
    'aardvark','abacus','abalone','admiral','african_elephant','albatross','alligator','alpaca','anaconda',
    'angelfish','ant','anteater','antelope','armadillo','axolotl','baboon','barracuda','basset_hound','bat',
    'beagle','bear','beetle','beluga_whale','bison','black_panther','black_widow_spider','blowfish','bobcat','buffalo',
    'bulldog','bullfrog','butterfly','coyote','chicken','chinchilla','chimpanzee','cockroach','cod','cougar','cow',
    'crab','crocodile','crow','deer','dingo','dodo','dog','dolphin','donkey','dove','duck','eagle','eel',
    'elephant','elk','emu','falcon','ferret','finch','flamingo','fly','fox','frog','gazelle','geese','gerbil',
    'giraffe','goat','goldfish','goose','gorilla','grasshopper','guinea_pig','hamster','hare','hawk','hippopotamus',
    'hornet','horse','hummingbird','hyena','ibex','iguana','impala','jaguar','jellyfish','kangaroo','koala','komodo_dragon',
    'kookaburra','ladybug','lamb','lark','leopard','lion','lizard','llama','lobster','locust','mongoose','monkey',
    'moose','moth','mountain_lion','mouse','octopus','orangutan','ostrich','otter','owl','ox','panda','parrot','peacock',
    'penguin','pig','pigeon','polar_bear','porcupine','quail','rabbit','raccoon','rat','raven','reindeer','rhinoceros',
    'roadrunner','rooster','salmon','scorpion','seal','shark','sheep','shrimp','skunk','slug','snail','snake','sparrow',
    'spider','squid','squirrel','starling','stingray','stoat','stork','swan','termite','toad','tortoise','toucan','tree_frog',
    'trout','tuna','turkey','turtle','vulture','wallaby','walrus','wasp','weasel','whale','wolf','wolverine','woodpecker',
    'zebra'
})

#Graph
df = pd.read_csv(DATA_FILE, sep='\t')
G_full = nx.Graph()
for _, row in df.iterrows():
    cue, resp = row['cue'], row['response']
    if isinstance(cue, str) and isinstance(resp, str):
        G_full.add_edge(cue.lower(), resp.lower(), weight=row['N'])

print(f"Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

animal_sub = G_full.subgraph(full_animals).copy()
communities = list(greedy_modularity_communities(animal_sub))

cluster_sizes = [len(c) for c in communities]
print(f"\nCluster size summary:")
print(f"Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, "
      f"Mean: {np.mean(cluster_sizes):.2f}, Median: {np.median(cluster_sizes):.2f}")

size_counts = collections.Counter(cluster_sizes)
print("\nCluster size distribution (size: count):")
for sz, ct in sorted(size_counts.items()):
    print(f"{sz}: {ct}")

min_size = 15
max_size = 50
filtered_communities = [c for c in communities if min_size <= len(c) <= max_size]
clusters_auto = {i: set(c) for i, c in enumerate(filtered_communities)}
print(f"\nFiltered: Kept {len(clusters_auto)} clusters with size between {min_size}–{max_size}")

def walk_by_time_per_cluster(G, clusters, start, steps, rho, irt_thresh):
    seen = set()
    cluster_ids = list(clusters.keys())
    cur_idx = 0
    cum_irt = 0.0
    cur = start

    cluster_counts = {cid: 0 for cid in cluster_ids}

    for t in range(steps):
        cid = cluster_ids[cur_idx]
        if cur in clusters[cid] and cur not in seen:
            seen.add(cur)
            cluster_counts[cid] += 1

        if cum_irt >= irt_thresh:
            cur_idx = (cur_idx + 1) % len(cluster_ids)
            cum_irt = 0.0

        if np.random.rand() < rho or cur not in G:
            cur = start
            continue

        nbrs = list(G[cur])
        if not nbrs:
            cur = start
            continue
        wts = np.array([G[cur][n]['weight'] for n in nbrs], float)
        probs = wts / wts.sum()
        nxt = np.random.choice(nbrs, p=probs)

        cum_irt += 1.0 / G[cur][nxt]['weight']
        cur = nxt

    return cluster_counts

results = []
for size in TEST_CLUSTER_SIZES:
    reduced_clusters = {
        cid: set(random.sample(list(mem), min(size, len(mem))))
        for cid, mem in clusters_auto.items()
    }

    total_counts = []
    per_cluster_lists = {cid: [] for cid in reduced_clusters}

    for _ in range(NUM_WALKERS):
        counts = walk_by_time_per_cluster(
            G_full, reduced_clusters, start='animal',
            steps=STEPS_PER_WALK, rho=RESTART_PROB, irt_thresh=IRT_THRESHOLD
        )
        total_counts.append(sum(counts.values()))
        for cid, cnt in counts.items():
            per_cluster_lists[cid].append(cnt)

    overall_mean = np.mean(total_counts)
    mean_per_cluster = np.mean([
        np.mean(per_cluster_lists[cid]) for cid in reduced_clusters
    ])

    print(f"\nCluster size {size}:")
    print(f"  Overall mean retrieved = {overall_mean:.2f}")
    print(f"  Mean per cluster       = {mean_per_cluster:.2f}")
    results.append((size, overall_mean, mean_per_cluster))

sizes, overall, per_cl = zip(*results)
plt.figure(figsize=(7,5))
plt.plot(sizes, overall, marker='o', label='Overall Mean Retrieved')
plt.plot(sizes, per_cl, marker='s', label='Mean per Cluster')
plt.xlabel("Artificial Cluster Size")
plt.ylabel("Mean Retrieved Animals")
plt.title("Retrieval vs. Cluster Size (Time‐Based IRT Switch)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cluster_sizes    = np.array([3,   5,    7,    10,   15])
retrieved_means  = np.array([0.08, 0.33, 0.36, 0.57, 0.78])

def power_law(x, a, b):
    return a * x**b

params, _ = curve_fit(power_law, cluster_sizes, retrieved_means)
a, b = params

x_fit = np.linspace(cluster_sizes.min(), cluster_sizes.max(), 200)
y_fit = power_law(x_fit, a, b)

# === Plot ===
plt.figure(figsize=(7, 5))
plt.plot(cluster_sizes, retrieved_means, 'o', label='Observed means')
plt.plot(x_fit, y_fit, 'r--', label=f'Fit: $y = {a:.3f}\\,x^{{{b:.3f}}}$')
plt.xlabel("Cluster Size")
plt.ylabel("Mean Retrieved Animals")
plt.title("Cluster Size vs. Retrieved Animals with Power-Law Fit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
vocab_known = np.array([10, 25, 50, 100, 166])
vocab_retrieved = np.array([1.87, 3.79, 7.87, 14.91, 26.32])

cluster_sizes = np.array([3, 5, 7, 10, 15])
cluster_mean_retrieved = np.array([0.08, 0.33, 0.36, 0.57, 0.78])

r_value, p_value = pearsonr(vocab_retrieved, cluster_mean_retrieved)

print(f"Pearson correlation: r = {r_value:.3f}, p = {p_value:.3f}")

reg = LinearRegression().fit(vocab_retrieved.reshape(-1, 1), cluster_mean_retrieved)
predicted = reg.predict(vocab_retrieved.reshape(-1, 1))

# Plotting
plt.figure(figsize=(6, 4))
plt.scatter(vocab_retrieved, cluster_mean_retrieved, color='blue', label='Observed means')
plt.plot(vocab_retrieved, predicted, color='red', linestyle='--', label=f'Fit: y = {reg.intercept_:.3f} + {reg.coef_[0]:.3f}x')
plt.title(f"Correlation between Vocab-Level and Cluster-Level Retrieval\nr = {r_value:.3f}, p = {p_value:.3f}")
plt.xlabel("Mean Retrieved Animals (Vocab-Level)")
plt.ylabel("Mean Retrieved Animals per Cluster")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cluster_sizes    = np.array([3, 5, 7, 10, 15])
retrieved_means  = np.array([0.08, 0.33, 0.36, 0.57, 0.78])

def power_law(x, a, b):
    return a * x**b

params, _ = curve_fit(power_law, cluster_sizes, retrieved_means)
a, b = params

x_fit = np.linspace(cluster_sizes.min(), cluster_sizes.max(), 200)
y_fit = power_law(x_fit, a, b)

residuals = retrieved_means - power_law(cluster_sizes, a, b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((retrieved_means - np.mean(retrieved_means))**2)
r_squared = 1 - (ss_res / ss_tot)

plt.figure(figsize=(7, 5))
plt.plot(cluster_sizes, retrieved_means, 'o', label='Observed means')
plt.plot(x_fit, y_fit, 'r--', label=f'Fit: y = {a:.3f}·x^{b:.3f}')
plt.xlabel("Cluster Size")
plt.ylabel("Mean Retrieved Animals")
plt.title(f"Cluster Size vs. Retrieved Animals with Power-Law Fit\nR² = {r_squared:.3f}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"R-squared: {r_squared:.3f}")

