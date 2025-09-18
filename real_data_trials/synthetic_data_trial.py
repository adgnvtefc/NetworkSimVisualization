import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim  as ns
from networkvis import NetworkVis as nv
from comparisons import Comparisons 
from plotting import plot_trials
import torch
import pickle

def save_graph_model(model, filename="graph_model.pth"):
    """
    Save the entire GraphQ model:
     - GNN weights
     - Optimizer state
     - Current epsilon
     - Etc.
    """
    checkpoint = {
        # The GCN part's parameters
        "gnn_state_dict": model.gnn.state_dict(),
        
        # The optimizer’s parameters
        "optimizer_state_dict": model.optimizer.state_dict(),
        
        # Additional fields you might want to store
        "epsilon": model.epsilon,
        "epsilon_decay": model.epsilon_decay,
        "epsilon_min": model.epsilon_min,
        "gamma": model.gamma,
        "lr": model.lr
    }
    
    torch.save(checkpoint, filename)
    print(f"GraphQ model saved to {filename}")




graph = ns.init_random_graph(200, 600, 1, 2)
print(graph)

# with open('test.gpickle', 'rb') as f:
#     graph = pickle.load(f)

# print(graph.nodes())
# print(graph.edges())

#with open('test.gpickle', 'wb') as f:
#    pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

pos = nx.spring_layout(graph)  # Positioning of nodes
algorithms = [
    'graph', 
    'dqn', 
    'whittle', 
    'hillclimb', 
    'none']
#ALL PREVIOUS EXPERIMENTS RAN WITH 30 ACTIONS
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL=5
comp = Comparisons()

g_model = comp.train_graph(graph, NUM_ACTIONS, CASCADE_PROB)
#save_graph_model(g_model, "graph_q_checkpoint.pth")

dqn_model = comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)
#torch.save(dqn_model.state_dict(), "dqn_model.pth")

comp.train_whittle(graph, GAMMA)

metadata = {"algorithms": algorithms,
            "initial_graph": graph,
            "num_comparisons": NUM_COMPARISONS,
            "num_actions": NUM_ACTIONS,
            "cascade_prob": CASCADE_PROB,
            "gamma": GAMMA,
            "timesteps": TIMESTEPS}

results = (comp.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=graph, 
    num_comparisons=NUM_COMPARISONS, 
    num_actions=NUM_ACTIONS,
    cascade_prob=CASCADE_PROB, 
    gamma=GAMMA, 
    timesteps=TIMESTEPS, 
    timestep_interval=TIMESTEP_INTERVAL))

plot_trials(
    results, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison",
    metadata=metadata)         # prefix for output filed

print(algorithms)
print(f"num comparisons: {NUM_COMPARISONS}")
print(f"num actions: {NUM_ACTIONS}")
print(f"cascade prob: {CASCADE_PROB}")
print(len(graph.nodes))
print(len(graph.edges))