import random
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from collections import deque

from game import SnakeGameAI, Direction, Point
from deep_qlearning_model import Linear_QNet, QTrainer
from nn_model import PolicyNet

# Base agent (state representation)
class BaseAgent:
    def get_state(self, game):
        """
        Returns an 11-dimensional binary vector representing the current state.
        The state includes danger detection (front, right, left), current direction,
        and the food location relative to the snake's head.
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Current direction (4 values)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to head (4 values)
            game.food.x < game.head.x,  # food is left
            game.food.x > game.head.x,  # food is right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y   # food is down
        ]
        return np.array(state, dtype=int)


# Q‑Learning Agent using a table (dictionary)
class TableQAgent(BaseAgent):
    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay factor per game
        self.epsilon = self.epsilon_start
        self.gamma = 0.9         # Discount factor for future rewards
        self.alpha = 0.1         # Learning rate for Q-value updates
        self.q_table = {}        # Q-table mapping state tuple to Q-values list
    
    def get_action(self, state):
        """
        Choose an action using an epsilon-greedy strategy.
        """
        final_move = [0, 0, 0]
        if random.randint(0,200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0, 0, 0]
            action = np.argmax(self.q_table[state_key])
        final_move[action] = 1
        return final_move

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q Learning update rule:
          Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0, 0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0, 0, 0]
        
        action_index = np.argmax(np.array(action))
        current_q = self.q_table[state_key][action_index]
        max_next_q = max(self.q_table[next_state_key])
        target = reward + (0 if done else self.gamma * max_next_q)
        self.q_table[state_key][action_index] = current_q + self.alpha * (target - current_q)
    
    def update_epsilon(self):
        self.n_games += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# SARSA Agent using a table (dictionary)
class SarsaQAgent(BaseAgent):
    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay factor per game
        self.epsilon = self.epsilon_start
        self.gamma = 0.9         # Discount factor for future rewards
        self.alpha = 0.1         # Learning rate for Q-value updates
        self.q_table = {}        # Q-table mapping state tuple to Q-values list

    def get_action(self, state):
        """
        Choose an action using an epsilon-greedy strategy.
        """
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0, 0, 0]
            action = np.argmax(self.q_table[state_key])
        final_move[action] = 1
        return final_move

    def update_sarsa(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table using the SARSA update rule:
          Q(s, a) = Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0, 0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0, 0, 0]
        
        action_index = np.argmax(np.array(action))
        next_action_index = np.argmax(np.array(next_action))
        current_q = self.q_table[state_key][action_index]
        # For terminal states, next Q is 0
        next_q = 0 if done else self.q_table[next_state_key][next_action_index]
        target = reward + self.gamma * next_q
        self.q_table[state_key][action_index] = current_q + self.alpha * (target - current_q)
    
    def update_epsilon(self):
        self.n_games += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
     
        
# Deep Q‑Learning Agent
class DeepQAgent(BaseAgent):
    def __init__(self, max_memory=100_000, batch_size=1000, lr=0.002):
        self.n_games = 0
        self.epsilon_start = 0.8   # Initial exploration rate
        self.epsilon_min = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.95 # Decay factor per game
        self.epsilon = self.epsilon_start
        self.gamma = 0.9         # Discount factor for future rewards
        self.memory = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.model = Linear_QNet(11, 256, 3)  # Network: input, hidden, output sizes
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)
    
    def get_action(self, state):
        """
        Choose an action using an epsilon-greedy strategy.
        """
        final_move = [0, 0, 0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train on a single experience.
        """
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """
        Train on a batch of experiences sampled from the replay memory.
        """
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def update_epsilon(self):
        self.n_games += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Neural Network Agent (Policy Gradient)
class NNAgent(BaseAgent):
    def __init__(self, lr=0.002, gamma=0.9):
        self.gamma = gamma
        self.policy_net = PolicyNet(11, 256, 3)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.reward_baseline = deque(maxlen=100)  # Baseline: last 100 episodes
    
    def get_action(self, state):
        """
        Given a state, compute the action probabilities and sample an action.
        The log probability of the chosen action is stored for later policy update.
        """
        state_tensor = torch.tensor(state, dtype=torch.float)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        final_move = [0, 0, 0]
        final_move[action.item()] = 1
        return final_move

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        """
        Update the policy network using the REINFORCE algorithm with a Baseline.
        """
        R = 0
        returns = []
        # Compute discounted returns (in reverse order)
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float)
        
        # Compute baseline as the moving average of past rewards
        baseline = np.mean(self.reward_baseline) if self.reward_baseline else 0
        returns = returns - baseline  # Reduce variance with baseline
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = 0
        entropy_loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss += -log_prob * R
            entropy_loss += -log_prob  # Approximation of entropy

        entropy_coef = 0.095  # Entropy coefficient to encourage exploration
        loss = policy_loss + entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the baseline with the latest episode's total reward
        self.reward_baseline.append(sum(self.rewards))

        # Clear stored log probabilities and rewards for next episode
        self.log_probs = []
        self.rewards = []