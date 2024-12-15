import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from env import GuanDanEnv
from itertools import combinations

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 200000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DQN Network
class HighLevelDQN(nn.Module):
    def __init__(self, input_dim, high_level_actions):
        super(HighLevelDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, high_level_actions)  # 输出单张、对子、三带、炸弹、Pass
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class LowLevelDQN(nn.Module):
    def __init__(self, input_dim, low_level_actions):
        super(LowLevelDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, low_level_actions)  # 输出具体牌型组合
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class HierarchicalDQNAgent:
    def __init__(self, input_dim, high_level_actions, low_level_actions):
        self.high_level_net = HighLevelDQN(input_dim, high_level_actions).to(DEVICE)
        self.low_level_net = LowLevelDQN(input_dim, low_level_actions).to(DEVICE)
        self.high_level_optimizer = optim.Adam(self.high_level_net.parameters(), lr=LR)
        self.low_level_optimizer = optim.Adam(self.low_level_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.high_level_actions = high_level_actions
        self.low_level_actions = low_level_actions
        self.epsilon = 0.1
    
    def select_high_level_action(self, state):
        state = state.to(DEVICE)
        if random.random() < self.epsilon:
            return random.randint(0, self.high_level_actions - 1)
        else:
            with torch.no_grad():
                return self.high_level_net(state).argmax().item()

    def select_low_level_action(self, state, legal_actions):
        state = state.to(DEVICE)
        if not legal_actions:
            return []
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            with torch.no_grad():
                q_values = self.low_level_net(state)
                legal_q_values = [q_values[i].cpu().item() for i in range(len(legal_actions))]
                return legal_actions[np.argmax(legal_q_values)]
            
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        q_values = self.low_level_net(states).gather(1, actions)
        next_q_values = self.low_level_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        self.low_level_optimizer.zero_grad()
        loss.backward()
        self.low_level_optimizer.step()

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
    
def get_legal_singles(player_hand):
    return [[card] for card in player_hand]

def get_legal_pairs(player_hand):
    pairs = []
    for combo in combinations(player_hand, 2):
        if combo[0] // 4 == combo[1] // 4:  # 检查点数是否相同
            pairs.append(list(combo))
    return pairs

def get_legal_triples(player_hand):
    triples = []
    for combo in combinations(player_hand, 3):
        if combo[0] // 4 == combo[1] // 4 == combo[2] // 4:  # 检查点数是否相同
            triples.append(list(combo))
    return triples

def get_legal_bombs(player_hand):
    bombs = []
    for combo in combinations(player_hand, 4):
        if combo[0] // 4 == combo[1] // 4 == combo[2] // 4 == combo[3] // 4:  # 检查点数是否相同
            bombs.append(list(combo))
    return bombs

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
    high_level_actions = 5  # 高层网络动作：单张、对子、三带、炸弹、Pass
    low_level_actions = 108  # 底层网络动作：从手牌中选择具体组合

    agent = HierarchicalDQNAgent(input_dim, high_level_actions, low_level_actions)
    
    # 加载之前的模型权重
    try:
        agent.high_level_net.load_state_dict(torch.load("high_level_dqn_guandan.pth"))
        agent.low_level_net.load_state_dict(torch.load("low_level_dqn_guandan.pth"))
        print("模型权重加载成功，继续训练。")
    except FileNotFoundError:
        print("未找到权重文件，将从头开始训练。")
        
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        current_player = 0
        print(f"Episode {episode}")
        while not done:
            player_hand = obs[current_player]['deck']
            state = torch.FloatTensor(encode_hand(player_hand)).to(DEVICE)
            # Step 1: High-Level action selection (e.g., Single, Pair, Triple, Bomb, Pass)
            high_level_action = agent.select_high_level_action(state)
            
            # 根据 High-Level 动作生成 Low-Level 合法动作
            if high_level_action == 0:
                legal_actions = get_legal_singles(player_hand)
            elif high_level_action == 1:
                legal_actions = get_legal_pairs(player_hand)
            elif high_level_action == 2:
                legal_actions = get_legal_triples(player_hand)
            elif high_level_action == 3:
                legal_actions = get_legal_bombs(player_hand)
            else:
                legal_actions = [[]]  # Pass 动作

            # Step 2: Low-Level action selection
            if legal_actions:
                action = agent.select_low_level_action(state, range(len(legal_actions)))
                chosen_action = legal_actions[action]
            else:
                chosen_action = []
                 
            
            
            # 执行选择的动作
            response = {'player': current_player, 'action': chosen_action, 'claim': chosen_action}
            print(f"Current Player: {current_player}, player_hand: {player_hand}, action: {chosen_action}")
            # Step后下一个状态
            next_obs = env.step(response)
            # 当前玩家出牌后的reward
            reward = env._get_obs(current_player)[current_player]['reward']
            done = env.done
            # 下一个玩家
            current_player = (current_player + 1) % 4
            player_hand = next_obs[current_player]['deck']
            next_state = torch.FloatTensor(encode_hand(player_hand)).to(DEVICE)
            obs = next_obs
            agent.memory.push(state.cpu(), high_level_action, reward, next_state.cpu(), done)
            # state = next_state
            total_reward += reward
            agent.update()

            

        # if episode % TARGET_UPDATE == 0:
        #     agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(agent.high_level_net.state_dict(), "high_level_dqn_guandan.pth")
    torch.save(agent.low_level_net.state_dict(), "low_level_dqn_guandan.pth")


if __name__ == "__main__":
    train()
