import gymnasium as gym
from algorithms.deep_q_env import NetworkInfluenceEnv
import random
import torch
import torch.nn as nn
import numpy as np
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import os
from dataclasses import dataclass
from typing import Dict
import time
from typing import Dict, Iterator, Tuple


log_path = os.path.join('logs', 'dqn')
writer = SummaryWriter(log_path)
writer.add_text("Experiment Info", "DQN training with custom environment")
logger = TensorboardLogger(writer)

class _NonNegLinear(nn.Module):
    """
    Linear layer with nonnegative weights and biases via softplus parameterization.
    Ensures monotonicity needed in DSF construction.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Unconstrained parameters
        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_param)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_param', None)

        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.softplus(self.weight_param)
        if self.bias_param is not None:
            bias = self.softplus(self.bias_param)
            return x @ weight.t() + bias
        return x @ weight.t()


class DSFQNet(nn.Module):
    """
    Deep Submodular Q Network (DSF) that is submodular and monotone in the action set A
    for any fixed state s. It follows the DSF design: nonnegative linear mixing between
    layers and concave, nondecreasing activations.

    Q(s, A) = w_out^T phi_L( ... phi_1( A ⊙ g(s) @ W1 + b1 ) @ W2 + b2 ... ) + b_out

    - We enforce Wk >= 0 and bk >= 0 via softplus.
    - phi_k(x) = 1 - exp(-x) which is concave and nondecreasing on x >= 0.
    - g(s) >= 0 is a state-dependent nonnegative gating over items (constant given s),
      so Q is submodular in A for each fixed s per DSF theory.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128), dropout_p: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # State-dependent nonnegative gating that produces per-item weights
        self.gate = _NonNegLinear(state_dim, action_dim)

        # DSF hidden layers operating on the (gated) action vector
        in_dim = action_dim
        layers = []
        alpha_params = []
        for h in hidden_sizes:
            layers.append(_NonNegLinear(in_dim, h))
            # Learnable positive alpha for activation scaling per layer
            alpha_param = nn.Parameter(torch.tensor(1.0))
            alpha_params.append(alpha_param)
            in_dim = h
        self.layers = nn.ModuleList(layers)
        self.alpha_params = nn.ParameterList(alpha_params)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()

        # Nonnegative readout to scalar
        self.readout = _NonNegLinear(in_dim, 1)

    @staticmethod
    def concave_activation_scaled(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # Ensure nonnegativity before applying; inputs should already be >= 0
        x = torch.relu(x)
        alpha_pos = torch.nn.functional.softplus(alpha) + 1e-6
        return (1.0 - torch.exp(-alpha_pos * x)) / alpha_pos

    def forward(self, state: torch.Tensor, action: torch.Tensor, state_shape=None, action_shape=None) -> torch.Tensor:
        # State-dependent per-item weights g(s) >= 0
        g = self.gate(state)  # [B, action_dim], nonnegative
        # Mask actions by state if desired: avoid credit for already active nodes
        # a_tilde remains in [0, inf) and is linear in action
        a_tilde = action * g

        h = a_tilde
        for lin, alpha in zip(self.layers, self.alpha_params):
            z = lin(h)
            h = self.concave_activation_scaled(z, alpha)
            h = self.dropout(h)

        q = self.readout(h)  # [B, 1]
        return q


class QNetBaseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetBaseline, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, state, action, state_shape=None, action_shape=None):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value

# Define the custom policy
class CustomQPolicy(BasePolicy):
    def __init__(self, model, optim, action_dim, k=5, gamma=0.95, epsilon=1.0, target_model: nn.Module = None, tau: float = 0.005):
        super().__init__(action_space=gym.spaces.MultiBinary(action_dim))
        self.model = model
        self.optim = optim
        self.k = k
        self.action_dim = action_dim
        self._gamma = gamma
        self.epsilon = epsilon  # Epsilon for epsilon-greedy exploration
        self.target_model = target_model
        if self.target_model is None:
            # If no target provided, mirror the current model
            import copy
            self.target_model = copy.deepcopy(self.model)
        self._tau = tau

    def forward(self, batch, state=None):
        obs = batch.obs  # Shape: [batch_size, state_dim]

        # Convert obs to PyTorch tensor
        device = next(self.model.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.float().to(device)

        batch_size = obs.shape[0]

        act = torch.zeros(batch_size, self.action_dim, device=device)

        for i in range(batch_size):
            base_state = obs[i].clone()
            selected_node_indices = []

            # Build the action set greedily by evaluating multi-hot sets
            for _ in range(self.k):
                # Available: nodes not active in state and not already selected
                all_available = (base_state == 0).nonzero(as_tuple=False).squeeze().tolist()
                if isinstance(all_available, int):
                    all_available = [all_available]
                # Exclude already-selected indices this step
                available_indices = [idx for idx in all_available if idx not in selected_node_indices]
                if not available_indices:
                    break

                # Form candidate action sets: current selected ∪ {idx}
                candidates = []
                for idx in available_indices:
                    action_vec = torch.zeros(self.action_dim, device=device)
                    if selected_node_indices:
                        action_vec[selected_node_indices] = 1
                    action_vec[idx] = 1
                    candidates.append(action_vec)
                actions_tensor = torch.stack(candidates)
                states_tensor = base_state.unsqueeze(0).repeat(len(available_indices), 1)

                with torch.no_grad():
                    q_values = self.model(states_tensor, actions_tensor).squeeze()

                if random.random() < self.epsilon:
                    selected_idx = random.choice(range(len(available_indices)))
                else:
                    selected_idx = torch.argmax(q_values).item()

                selected_node = available_indices[selected_idx]
                selected_node_indices.append(selected_node)

            act[i, selected_node_indices] = 1

        return Batch(act=act)

    def learn(self, batch, **kwargs):
        # Convert batch data to tensors
        device = next(self.model.parameters()).device

        states = batch.obs
        actions = batch.act
        rewards = batch.rew
        next_states = batch.obs_next
        dones = batch.done

        # Convert to tensors if necessary
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).view(-1)

        self.optim.zero_grad()
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()
        self.optim.step()

        return TrainStepResult(loss=loss.item())


    def compute_loss(self, states, actions, rewards, next_states, dones):
        gamma = self._gamma

        # Flatten actions for input
        actions = actions.view(-1, self.action_dim)
        states = states.view(-1, self.action_dim)

        q_values = self.model(states, actions).squeeze()

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            next_q_values = []
            for i in range(next_states.shape[0]):
                next_state = next_states[i]
                available_indices = (next_state == 0).nonzero(as_tuple=False).squeeze().tolist()
                if not available_indices:
                    max_q_value = 0.0
                else:
                    if isinstance(available_indices, int):
                        available_indices = [available_indices]

                    actions_list = []
                    for idx in available_indices:
                        action = torch.zeros(self.action_dim, device=next_state.device)
                        action[idx] = 1
                        actions_list.append(action)
                    actions_tensor = torch.stack(actions_list)
                    states_tensor = next_state.unsqueeze(0).repeat(len(available_indices), 1)

                    # Main network selects argmax
                    q_vals_main = self.model(states_tensor, actions_tensor).squeeze()
                    best_idx = torch.argmax(q_vals_main).item()
                    best_action = actions_tensor[best_idx].unsqueeze(0)
                    best_state = states_tensor[best_idx].unsqueeze(0)
                    # Target network evaluates
                    max_q_value = self.target_model(best_state, best_action).item()
                next_q_values.append(max_q_value)
            next_q_values = torch.tensor(next_q_values, device=states.device)

            target_q_values = rewards + gamma * (1 - dones) * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        # Soft-update target network
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.mul_(1.0 - self._tau)
                target_param.data.add_(self._tau * param.data)
        return loss

@dataclass
class TrainStepResult:
    loss: float

    def get_loss_stats_dict(self) -> Dict[str, float]:
        return {'loss': self.loss}

    def keys(self) -> Tuple[str, ...]:
        return ("loss",)

    def __getitem__(self, key: str) -> float:
        if key == "loss":
            return self.loss
        raise KeyError(f"TrainStepResult does not contain key: {key}")

    def __setitem__(self, key: str, value: float) -> None:
        if key == "loss":
            self.loss = value
        else:
            raise KeyError(f"TrainStepResult does not contain key: {key}")
    
    def items(self) -> Iterator[Tuple[str, float]]:
        for k in self.keys():
            yield k, self[k]
    


def train_dqn_agent(config, num_actions, num_epochs=3):
    start_time = time.perf_counter()

    # Set up environment
    def get_env():
        return NetworkInfluenceEnv(config)
    
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        epsilon = max(0.1, 1 - env_step / 50000)  # Linear decay
        policy.epsilon = epsilon
    def test_fn(epoch, env_step):
        pass

    # Instantiate the model and policy
    state_dim = config['num_nodes']
    action_dim = config['num_nodes']

    use_dsf = config.get('use_dsf', True)
    if use_dsf:
        model = DSFQNet(state_dim, action_dim)
    else:
        model = QNetBaseline(state_dim, action_dim)
    target_model = type(model)(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    policy = CustomQPolicy(model, optimizer, action_dim=action_dim, k=num_actions, gamma=0.99, target_model=target_model, tau=0.005)

    # Set up collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=20000 * train_envs.env_num, buffer_num=train_envs.env_num))
    test_collector = Collector(policy, test_envs)

    # Start training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=num_epochs,
        step_per_epoch=3000,
        step_per_collect=50,
        episode_per_test=0,
        batch_size=64,
        update_per_step=0.1,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        logger=logger
    ).run()

    end_time = time.perf_counter()
    
    time_taken = end_time - start_time
    train_dqn_agent.time = time_taken

    return model, policy

train_dqn_agent.time = 0
def get_train_dqn_agent_time():
    return train_dqn_agent.time

def select_action_dqn(graph, model, num_actions):
    """
    Hill-climbing action selection using a DQN model.
    
    Args:
        graph: The graph representing the environment.
        model: The trained DQN model.
        num_actions: The number of actions to select (k).

    Returns:
        A list of node objects representing the selected actions.
    """
    start_time = time.perf_counter()
    select_action_dqn.times_called += 1
    num_nodes = len(graph.nodes())
    state = np.array([int(graph.nodes[i]['obj'].isActive()) for i in graph.nodes()], dtype=np.float32)
    device = next(model.parameters()).device
    selected_node_indices = []

    # Greedy selection on multi-hot action sets evaluated by the DSF model
    base_state = state.copy()
    for _ in range(num_actions):
        state_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=device).unsqueeze(0)

        # Build candidate sets by adding each feasible node to current selection
        candidates = []
        candidate_indices = []
        active_nodes_indices = [i for i, node in enumerate(graph.nodes()) if graph.nodes[node]['obj'].isActive()]
        forbidden = set(active_nodes_indices) | set(selected_node_indices)
        for idx in range(num_nodes):
            if idx in forbidden:
                continue
            action_vec = torch.zeros(num_nodes, device=device)
            if selected_node_indices:
                action_vec[selected_node_indices] = 1
            action_vec[idx] = 1
            candidates.append(action_vec)
            candidate_indices.append(idx)

        if not candidates:
            break

        actions_tensor = torch.stack(candidates)
        states = state_tensor.repeat(actions_tensor.size(0), 1)

        with torch.no_grad():
            q_values = model(states, actions_tensor).squeeze()

        top_local = torch.argmax(q_values).item()
        top_action_index = candidate_indices[top_local]
        selected_node_indices.append(top_action_index)
    
    seeded_nodes = [graph.nodes[node_index]['obj'] for node_index in selected_node_indices]
    end_time = time.perf_counter()
    elapsed = end_time - start_time
        
    select_action_dqn.total_time += elapsed
    return seeded_nodes
select_action_dqn.total_time = 0.0
select_action_dqn.times_called = 0

@staticmethod
def get_dqn_total_time():
    return select_action_dqn.total_time
@staticmethod
def get_dqn_times_called():
    return select_action_dqn.times_called