import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
import networkx as nx
import random

class NetworkInfluenceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config, render_mode=None):
        super(NetworkInfluenceEnv, self).__init__()
        self.config = config
        # Define action and observation space
        self.num_nodes = config['num_nodes']
        self.num_actions = config['num_nodes']
        self.cascade_prob = config['cascade_prob']
        self.stop_percent = config['stop_percent']
        self.reward_function = config['reward_function']
        self.gamma = 0.8  # Discount factor for the enhanced reward function

        self.observation_space = spaces.MultiBinary(self.num_nodes)
        self.action_space = spaces.MultiBinary(self.num_nodes)

        self.latest_step_reward = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize your network here
        self.original_graph = config['graph']
        self.pos = nx.spring_layout(self.original_graph)

        self.graph = config['graph']
        self.state = np.zeros(self.num_nodes, dtype=np.int8)

    def _get_obs(self):
        # Return the state as a NumPy array
        return self.state.copy()

    def _get_info(self):
        # if self.reward_function == "normal":
        #     return {"reward": ns.reward_function(graph=self.graph, seed=None, cascade_reward=self.cascade_prob)}
        
        # return {
        #     "reward": ns.enhanced_reward_function(
        #         graph=self.graph,
        #         seed=None,
        #         action_size=self.num_actions,
        #         gamma=self.gamma,
        #         cascade_reward=self.cascade_prob
        #     )
        # }
        return {"reward": self.latest_step_reward}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        # Reset the network to the initial state
        self.graph = copy.deepcopy(self.original_graph)
        self.state = np.zeros(self.num_nodes, dtype=np.int8)
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        # cur_reward = ns.reward_function(
        #      graph=self.graph,
        #      seed=None,
        #      cascade_reward=self.cascade_prob
        # )

        action_indices = np.argwhere(action).flatten()

        # expected_future_reward = 0

        # for _ in range(10):
        #     new_graph = ns.simulate_next_state(self.graph, action=action_indices)
        #     expected_future_reward += ns.enhanced_reward_function(graph=new_graph, seed=None, action_size=self.num_actions, gamma=self.gamma, cascade_reward=self.cascade_prob)

        # ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        # ns.active_state_transition_graph_indices(self.graph, action_indices)
        # ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        # ns.rearm_nodes(self.graph)

        # self.state = np.array([
        #     int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()
        # ])

        # reward = cur_reward + self.gamma * (expected_future_reward / 10)

        #horizon=2 is too much for this laptop to handle
        reward = ns.action_value_function(self.graph, action, num=1, gamma=self.gamma, horizon=1, num_samples=1)
        self.latest_step_reward = reward

        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        self.state = np.array([
            int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()
        ])

        terminated = (
            sum([self.graph.nodes[i]['obj'].isActive() for i in self.graph.nodes()]) >= self.stop_percent * self.num_nodes
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        if self.render_mode == "human":
            nv.render(self.graph, self.pos)

# Register the environment
from gymnasium.envs.registration import register

register(
    id='NetworkInfluence-v0',
    entry_point='network_env:NetworkInfluenceEnv',
    max_episode_steps=300,
)
