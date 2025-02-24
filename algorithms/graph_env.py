import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import copy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns

def convert_nx_to_pyg(G: nx.Graph, last_action_indices=None) -> Data:
    """
    Convert a NetworkX graph to a PyG Data object.
    If last_action_indices is provided, we'll use it to set
    the 7th feature = 1 for the node(s) that were acted on.
    """
    for n in G.nodes():
        node_obj = G.nodes[n]['obj']

        # Check if this node was in the action list
        if last_action_indices is not None and n in last_action_indices:
            was_acted_on = 1.0
        else:
            was_acted_on = 0.0

        # Create each node's feature vector (now length 7)
        G.nodes[n]['x'] = torch.tensor([
            float(node_obj.isActive()),
            node_obj.getValue(),
            node_obj.active_activation_active,
            node_obj.active_activation_passive,
            node_obj.passive_activation_active,
            node_obj.passive_activation_passive,
            was_acted_on,  # 7th feature
        ], dtype=torch.float)

    data = from_networkx(G)
    return data

class GraphEnv(gym.Env):
    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.num_nodes = config['num_nodes']
        self.cascade_prob = config['cascade_prob']
        self.stop_percent = config['stop_percent']
        self.reward_function = config['reward_function']
        self.gamma = 0.8
        self.reward_mult = 1

        self.original_graph = config['graph']
        self.latest_step_reward = 0

        # We'll store the most recent list of activated node indices here
        self.last_action_indices = None

        # Action space: pick which node to activate (single choice for now)
        self.action_space = spaces.Discrete(n=self.num_nodes, start=0)

        # Observation space: now shape=(num_nodes, 7) for x
        self.observation_space = spaces.Dict({
            "x": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_nodes, 7),
                dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0,
                high=self.num_nodes - 1,
                shape=(2, self.original_graph.number_of_edges() * 2),
                dtype=np.int64
            )
        })

        self.graph = None
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.graph = copy.deepcopy(self.original_graph)
        self.state = np.zeros(self.num_nodes, dtype=np.int8)
        self.reward_mult = 1

        # No last action at the beginning
        self.last_action_indices = None

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        # Convert to PyG Data, pass last_action_indices so we
        # can set the 7th feature in node attributes
        pyg_data = convert_nx_to_pyg(self.graph, self.last_action_indices)
        return {
            "x": pyg_data.x.numpy(),             # shape: (num_nodes, 7)
            "edge_index": pyg_data.edge_index.numpy()  # shape: (2, num_edges)
        }

    def _get_info(self):
        return {"reward": self.latest_step_reward}

    def step(self, action):
        # If you're taking only a single action, wrap it in a list
        action_indices = [action]

        reward = ns.action_value_function(
            self.graph, action,
            num_actions=1,
            cascade_prob=self.cascade_prob,
            gamma=self.gamma,
            horizon=1,
            num_samples=1
        )
        reward *= self.reward_mult
        self.reward_mult -= 0.02
        self.latest_step_reward = reward

        # Update graph state
        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        # Update which nodes are active
        self.state = np.array([
            int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()
        ])

        # Store the current step's action_indices so _get_obs() can show it
        self.last_action_indices = action_indices

        # Check termination condition
        terminated = (
            sum(self.state) >= self.stop_percent * self.num_nodes
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info
