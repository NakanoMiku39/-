import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from env import GuanDanEnv

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 1000

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Agent
class DQNAgent:
    def __init__(self, input_dim, action_dim):
        self.policy_net = DQN(input_dim, action_dim)
        self.target_net = DQN(input_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.action_dim = action_dim
        self.steps_done = 0
        self.epsilon = 0.1
    
    def select_action(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                legal_q_values = q_values[legal_actions]
                return legal_actions[legal_q_values.argmax().item()]
    
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量并确保形状一致
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Get legal actions (simplified example)
def get_legal_actions(player_deck):
    legal_actions = [0]  # 0 represents 'Pass'
    if len(player_deck) > 0:
        for i in range(len(player_deck)):
            legal_actions.append(i + 1)  # Each index + 1 represents a card to play
    return legal_actions

def encode_hand(player_hand):
    """
    将玩家的手牌编码成 108 维向量。

    Args:
        player_hand (list): 玩家手牌，包含牌的编号。

    Returns:
        np.array: 108 维的 one-hot 编码向量。
    """
    encoded = np.zeros(108, dtype=np.float32)
    for card in player_hand:
        encoded[card] = 1
    return encoded

# Training Loop
def train():
    env = GuanDanEnv()
    input_dim = 108  # 输入维度改为 108
    action_dim = 28  # 包括 Pass 和 27 种出牌动作
    agent = DQNAgent(input_dim, action_dim)

    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        current_player = 0

        while not done:
            player_hand = obs[current_player]['deck']
            state = torch.FloatTensor(encode_hand(player_hand))
            legal_actions = get_legal_actions(player_hand)
            action = agent.select_action(state, legal_actions)

            if action == 0:
                response = {'player': current_player, 'action': [], 'claim': []}  # Pass
            else:
                card_to_play = player_hand[action - 1]
                response = {'player': current_player, 'action': [card_to_play], 'claim': [card_to_play]}

            # Step后下一个状态
            next_obs = env.step(response)
            # 当前玩家出牌后的reward
            reward = env._get_obs(current_player)[current_player]['reward']
            done = env.done
            # 下一个玩家
            current_player = (current_player + 1) % 4
            player_hand = next_obs[current_player]['deck']
            next_state = torch.FloatTensor(encode_hand(player_hand))
            obs = next_obs
            agent.memory.push(state, action, reward, next_state, done)
            # state = next_state
            total_reward += reward
            agent.update()

            

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(agent.policy_net.state_dict(), "dqn_guandan.pth")

if __name__ == "__main__":
    train()
