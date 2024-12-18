import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from env import GuanDanEnv
from itertools import combinations
import matplotlib.pyplot as plt
import gc
from utils import Utils

# Hyperparameters
NUM_AGENTS = 4
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 256
MEMORY_SIZE = 2000
TARGET_UPDATE = 10
EPISODES = 500000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils = Utils()
# DEVICE = torch.device("cpu")

print(f"Using {DEVICE}")
print(torch.cuda.is_available())  # 如果返回 False，表示 GPU 不可用
print(torch.cuda.device_count())  # 查看可用的 GPU 数量
print(torch.cuda.current_device())  # 获取当前 GPU 设备编号
print(torch.cuda.get_device_name(0))  # 查看 GPU 的名称（如果存在）
print(torch.__version__)
print(torch.version.cuda)
    
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
    def __init__(self):
        super(LowLevelDQN, self).__init__()
        self.fc1 = nn.Linear(324, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 108)  # 输出108维向量，每个维度对应一张牌的概率

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # 使用 sigmoid 确保输出在 [0, 1] 范围内

class HierarchicalDQNAgent:
    def __init__(self, input_dim, high_level_actions, low_level_actions):
        self.high_level_net = HighLevelDQN(input_dim, high_level_actions).to(DEVICE)
        self.high_level_target_net = HighLevelDQN(input_dim, high_level_actions).to(DEVICE)
        self.low_level_net = LowLevelDQN().to(DEVICE)
        self.low_level_target_net = LowLevelDQN().to(DEVICE)

        self.high_level_optimizer = optim.Adam(self.high_level_net.parameters(), lr=LR)
        self.low_level_optimizer = optim.Adam(self.low_level_net.parameters(), lr=LR)

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.high_level_actions = high_level_actions
        self.low_level_actions = low_level_actions
        self.epsilon = 0.5

        # 初始化目标网络权重
        self.high_level_target_net.load_state_dict(self.high_level_net.state_dict())
        self.low_level_target_net.load_state_dict(self.low_level_net.state_dict())

    def select_high_level_action(self, state):
        state = state.to(DEVICE)
        if random.random() < self.epsilon:
            return random.randint(0, self.high_level_actions - 1)
        else:
            with torch.no_grad():
                return self.high_level_net(state).argmax().item()

    def select_low_level_action(self, state, hand, high_level_action):
        """
        根据手牌和高层决策选择具体的牌组合。

        Args:
            hand (list): 当前手牌。
            high_level_action (int): 高层决策的动作索引。

        Returns:
            list: 选择的牌组合。
        """
        state = state.to(DEVICE)

        with torch.no_grad():
            q_values = self.low_level_net(state).cpu().numpy()

        # 筛选当前手牌中的牌，并根据 Q 值排序
        hand_q_values = [(card, q_values[card]) for card in hand]
        hand_q_values.sort(key=lambda x: x[1], reverse=True)

        # 根据高层决策选择相应的牌型
        if high_level_action == 0:  # 单张
            return utils.get_legal_singles([card for card, _ in hand_q_values])
        elif high_level_action == 1:  # 对子
            pairs = utils.get_legal_pairs([card for card, _ in hand_q_values], False)
            if pairs:
                return pairs
        elif high_level_action == 2:  # 三同张
            triples = utils.get_legal_triples([card for card, _ in hand_q_values], False)
            if triples:
                return triples
        elif high_level_action == 3:  # 炸弹
            bombs = utils.get_legal_bombs([card for card, _ in hand_q_values])
            if bombs:
                return bombs
        elif high_level_action == 4:  # 三连对（木板）
            triple_pairs = utils.get_legal_triple_pairs([card for card, _ in hand_q_values])
            if triple_pairs:
                return triple_pairs
        elif high_level_action == 5:  # 三同连张（钢板）
            triple_straights = utils.get_legal_triple_straight([card for card, _ in hand_q_values])
            if triple_straights:
                return triple_straights
        elif high_level_action == 6:  # 三带二（夯）
            three_with_pair = utils.get_legal_three_with_pair([card for card, _ in hand_q_values])
            if three_with_pair:
                return three_with_pair
        elif high_level_action == 7:  # 顺子
            straights = utils.get_legal_straights([card for card, _ in hand_q_values])
            if straights:
                return straights
        elif high_level_action == 8:  # 同花顺
            straight_flushes = utils.get_legal_straight_flushes([card for card, _ in hand_q_values])
            if straight_flushes:
                return straight_flushes
        elif high_level_action == 9:  # 火箭（王炸）
            rockets = utils.get_legal_rockets([card for card, _ in hand_q_values])
            if rockets:
                return rockets

        # 如果没有找到合适的组合，则 Pass
        return []

    def update_target_networks(self):
        self.high_level_target_net.load_state_dict(self.high_level_net.state_dict())
        self.low_level_target_net.load_state_dict(self.low_level_net.state_dict())

    def update(self, global_memory):
        if len(global_memory) < BATCH_SIZE:
            return 0, 0, 0, 0

        # 从记忆库中采样
        batch = global_memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 解包高层和低层动作
        high_level_actions, low_level_actions = zip(*actions)   
            
        low_level_actions_one_hot = []
        utils.Action2Onehot(low_level_actions, low_level_actions_one_hot)

        # 转换为 PyTorch 张量
        low_level_actions_one_hot = torch.FloatTensor(low_level_actions_one_hot).to(DEVICE)
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        high_level_actions = torch.LongTensor(high_level_actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # 更新低层网络
        low_level_loss, low_level_q_value = self._update_low_level_network(states, low_level_actions_one_hot, rewards, next_states, dones)

        # 更新高层网络
        high_level_loss, high_level_q_value = self._update_high_level_network(states, high_level_actions, rewards, next_states, dones)
        
        return low_level_loss, low_level_q_value, high_level_loss, high_level_q_value
    
    def _update_low_level_network(self, states, actions_one_hot, rewards, next_states, dones):
        q_values = (self.low_level_net(states) * actions_one_hot).sum(dim=1, keepdim=True)
        next_q_values = self.low_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        # self.low_level_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.low_level_optimizer.step()
        
        return loss.item(), q_values.mean().item()

    def _update_high_level_network(self, states, actions, rewards, next_states, dones):
        q_values = self.high_level_net(states).gather(1, actions)
        next_q_values = self.high_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        # self.high_level_optimizer.zero_grad(set_to_none=True)  # 清理梯度缓存
        loss.backward()
        self.high_level_optimizer.step()
        
        return loss.item(), q_values.mean().item()

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
    
def encode_obs(obs):
    """
    将观察结果编码为向量，包括手牌、上一次的出牌和历史出牌记录。

    Args:
        obs (dict): 单个玩家的观察结果。

    Returns:
        np.array: 编码后的观察向量。
    """
    encoded = np.zeros(108 + 108 + 108, dtype=np.float32)  # 手牌 + 上次出牌 + 历史出牌

    # 编码手牌
    for card in obs['deck']:
        encoded[card] = 1

    # 编码上次出牌
    for card in obs['last_move']['action']:
        encoded[108 + card] = 1

    # 编码历史出牌记录
    history_cards = []
    for move in obs['history']:
        history_cards.extend(move['action'])
    for card in history_cards:
        encoded[216 + card] = 1

    return encoded

# Training Loop
def train():
    env = GuanDanEnv()
    global_memory = ReplayMemory(MEMORY_SIZE)  # 全局共享经验池
    input_dim = 324  # 手牌 + 上次出牌 + 历史出牌记录
    high_level_actions = 11  # 高层网络动作：单张、对子、三带、炸弹、Pass...
    low_level_actions = 108  # 底层网络动作：从手牌中选择具体组合
    card_mapping = [utils.ChineseNum2Poker(i) for i in range(108)]
    agents = [HierarchicalDQNAgent(input_dim, high_level_actions, low_level_actions) for _ in range(NUM_AGENTS)]
    
    # 初始化指标记录
    agent_metrics = [{'low_level_loss': [], 'high_level_loss': [], 'reward': [], 'low_level_q_value': [], 'high_level_q_value': []} for _ in range(NUM_AGENTS)]
    
    high_level_action_space = ["单张", "对子", "三带", "炸弹", "三连对（木板）", "三同连张（钢板）", "三带二（夯）", "顺子", "同花顺", "火箭（王炸）", "过"]
    wins = 0

    agent = HierarchicalDQNAgent(input_dim, high_level_actions, low_level_actions)
    
    print(f"Validate cardmapping: {card_mapping}")
    # 加载之前的模型权重
    try:
        # agent.high_level_net.load_state_dict(torch.load("targetDQN/high_level_dqn_guandan.pth", map_location=DEVICE))
        # agent.low_level_net.load_state_dict(torch.load("targetDQN/low_level_dqn_guandan.pth", map_location=DEVICE))
        for i, agent in enumerate(agents):
            agent.high_level_net.load_state_dict(torch.load(f"targetDQN/agent_{i}_high_level.pth", map_location=DEVICE))
            agent.low_level_net.load_state_dict(torch.load(f"targetDQN/agent_{i}_low_level.pth", map_location=DEVICE))
        print("模型权重加载成功，继续训练。")
    except FileNotFoundError:
        print("未找到权重文件，将从头开始训练。")
        
    for episode in range(EPISODES):
        obs = env.reset()
        # done = False
        total_reward = 0
        current_player = 0
        episode_rewards = [0] * NUM_AGENTS  # 每个智能体的回合总奖励
        # print(f"Episode {episode}")
        while not env.done:
            agent = agents[current_player]
            player_obs = obs[current_player]
            # 如果当前玩家出完牌了
            if player_obs['status'] == "Finished":
                current_player = (current_player + 1) % 4
                continue
            
            state = torch.FloatTensor(encode_obs(player_obs)).to(DEVICE)
            # Step 1: High-Level action selection (e.g., Single, Pair, Triple, Bomb, Pass)
            high_level_action = agent.select_high_level_action(state)

            # Step 2: Low-Level action selection
            if high_level_action in range(high_level_actions):
                low_level_action = agent.select_low_level_action(state, player_obs['deck'], high_level_action)
            else:
                low_level_action = []

            if episode % 1000 == 0:
                print(f"Player {current_player}: {[card_mapping[i] for i in player_obs['deck']]}")
                print(f"high level action: {high_level_action_space[high_level_action]}")
                print(f"low level action: {low_level_action}")
                print(f"low level action: {[card_mapping[i] for i in low_level_action]}")
                # print(f"Selected action index: {action}, Legal actions length: {len(legal_actions)}")

            # 执行选择的动作
            response = {'player': current_player, 'action': low_level_action, 'claim': low_level_action}
            # Step后下一个状态
            next_obs = env.step(response)
            obs = next_obs
            # print(f"Next_obs: {next_obs}")
            # 当前玩家出牌后的reward
            reward = env._get_obs(current_player)[current_player]['reward']
            # total_reward += reward
            
            episode_rewards[current_player] += reward  # 累加当前玩家的奖励
            low_level_loss, low_level_q_value, high_level_loss, high_level_q_value = agent.update(global_memory)
            agent_metrics[current_player]['low_level_loss'].append(low_level_loss)
            agent_metrics[current_player]['low_level_q_value'].append(low_level_q_value)
            agent_metrics[current_player]['high_level_loss'].append(high_level_loss)
            agent_metrics[current_player]['high_level_q_value'].append(high_level_q_value)

            # 处理游戏结束时的 next_state
            if env.done:
                if env.game_state_info == "Finished":
                    print("A successful end game!")
                    win += 1
                
                    # print("A faulty game")
                next_state = torch.zeros_like(state)  # 游戏结束时，设置零向量
            else:
                # 下一个玩家
                current_player = next(iter(next_obs.keys()))
                # print(f"current_player: {current_player} \nnext_obs: {next_obs}")
                next_state = torch.FloatTensor(encode_obs(next_obs[current_player])).to(DEVICE)
                # else:
                #     print(f"Warning: next_obs does not contain data for player {current_player}")
                #     next_state = torch.zeros_like(state)  # 使用默认零向量
                

            # 存储经验
            # agent.memory.push(state.cpu(), (high_level_action, low_level_action), reward, next_state.cpu(), done)
            global_memory.push(state.cpu(), (high_level_action, low_level_action), reward, next_state.cpu(), env.done)
            
            
        # 记录奖励
        # reward_history.append(total_reward)
        # 每个 agent 记录总奖励
        for i in range(NUM_AGENTS):
            agent_metrics[i]['reward'].append(episode_rewards[i])
            
        if episode % TARGET_UPDATE == 0:
            for agent in agents:
                agent.update_target_networks()
            
        if episode % 1000  == 0:
            # print(f"Episode {episode}, Total Reward: {total_reward}")
            print(f"Episode {episode}")
            print("---")
            
    # 绘制权重变化
    # plot_training_curves(loss_history, reward_history, weight_history)
    
    
    # 保存权重
    for i, agent in enumerate(agents):
        torch.save(agent.high_level_net.state_dict(), f"targetDQN/agent_{i}_high_level.pth")
        torch.save(agent.low_level_net.state_dict(), f"targetDQN/agent_{i}_low_level.pth")

    print(f"Results:\nTotal episodes: {EPISODES}\nSuccessful end game: {wins}")
    utils.plot_agent_metrics(agent_metrics, "targetDQN/training_curves.png")
    

if __name__ == "__main__":
    train()
    

