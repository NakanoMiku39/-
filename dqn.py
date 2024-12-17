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
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 256
MEMORY_SIZE = 20000
TARGET_UPDATE = 10
EPISODES = 200000
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
        self.epsilon = 0.1

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

        # 根据高层决策选择相应数量的牌
        if high_level_action == 0:  # 单张
            # print(f"单张Hand Q-Values: {hand_q_values}")
            return [hand_q_values[0][0]]
        elif high_level_action == 1:  # 对子
            for i in range(len(hand_q_values) - 1):
                if hand_q_values[i][0] // 4 == hand_q_values[i + 1][0] // 4:
                    return [hand_q_values[i][0], hand_q_values[i + 1][0]]
        elif high_level_action == 2:  # 三带
            for i in range(len(hand_q_values) - 2):
                if (hand_q_values[i][0] // 4 == hand_q_values[i + 1][0] // 4 ==
                    hand_q_values[i + 2][0] // 4):
                    return [hand_q_values[i][0], hand_q_values[i + 1][0], hand_q_values[i + 2][0]]
        elif high_level_action == 3:  # 炸弹
            # print(f"炸弹Hand Q-Values: {hand_q_values}")
            for i in range(len(hand_q_values) - 3):
                if (hand_q_values[i][0] // 4 == hand_q_values[i + 1][0] // 4 ==
                    hand_q_values[i + 2][0] // 4 == hand_q_values[i + 3][0] // 4):
                    return [hand_q_values[i][0], hand_q_values[i + 1][0],
                            hand_q_values[i + 2][0], hand_q_values[i + 3][0]]

        # 如果没有找到合适的组合，则 Pass
        return []


            
    def update_target_networks(self):
        self.high_level_target_net.load_state_dict(self.high_level_net.state_dict())
        self.low_level_target_net.load_state_dict(self.low_level_net.state_dict())

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # 从记忆库中采样
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 解包高层和低层动作
        high_level_actions, low_level_actions = zip(*actions)   
            
        low_level_actions_one_hot = []
        utils.Action2Onehot(low_level_actions, low_level_actions_one_hot)
        # for action in low_level_actions:
        #     one_hot = np.zeros(108, dtype=np.float32)
        #     if action and action[0] != -1:
        #         for card in action:
        #             one_hot[card] = 1
        #     low_level_actions_one_hot.append(one_hot)

        # 转换为 PyTorch 张量
        low_level_actions_one_hot = torch.FloatTensor(low_level_actions_one_hot).to(DEVICE)
        # 使用 pad_sequence 对 low_level_actions 进行填充

        # print(f"High Level Actions: {high_level_actions}")
        # print(f"Low Level Actions: {low_level_actions}")
        # print(f"Low Level Actions One Hot: {low_level_actions_one_hot}")
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        high_level_actions = torch.LongTensor(high_level_actions).unsqueeze(1).to(DEVICE)
        # low_level_actions = torch.LongTensor(low_level_actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # 更新低层网络
        self._update_low_level_network(states, low_level_actions_one_hot, rewards, next_states, dones)

        # 更新高层网络
        self._update_high_level_network(states, high_level_actions, rewards, next_states, dones)

    # def _update_low_level_network(self, states, actions, rewards, next_states, dones):
    #     valid_indices = (actions != -1).all(dim=1)
    #     if valid_indices.any():
    #         valid_states = states[valid_indices]
    #         valid_actions = actions[valid_indices]
    #         valid_rewards = rewards[valid_indices]
    #         valid_next_states = next_states[valid_indices]
    #         valid_dones = dones[valid_indices]

    #         # 计算 Q 值
    #         q_values = self.low_level_net(valid_states).gather(1, valid_actions[:, 0].unsqueeze(1))
    #         next_q_values = self.low_level_target_net(valid_next_states).max(1)[0].unsqueeze(1)
    #         target_q_values = valid_rewards + GAMMA * next_q_values * (1 - valid_dones)

    #         loss = F.mse_loss(q_values, target_q_values)

    #         self.low_level_optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         self.low_level_optimizer.step()
    
    def _update_low_level_network(self, states, actions_one_hot, rewards, next_states, dones):
        q_values = (self.low_level_net(states) * actions_one_hot).sum(dim=1, keepdim=True)
        next_q_values = self.low_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        # self.low_level_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.low_level_optimizer.step()

    def _update_high_level_network(self, states, actions, rewards, next_states, dones):
        q_values = self.high_level_net(states).gather(1, actions)
        next_q_values = self.high_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        # self.high_level_optimizer.zero_grad(set_to_none=True)  # 清理梯度缓存
        loss.backward()
        self.high_level_optimizer.step()

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
    
# # 单张
# def get_legal_singles(player_hand):
#     return [[card] for card in player_hand]

# # 对子
# def get_legal_pairs(player_hand):
#     pairs = []
#     for combo in combinations(player_hand, 2):
#         if combo[0] // 4 == combo[1] // 4:
#             pairs.append(list(combo))
#     return pairs

# # 三连对（木板）
# def get_legal_triple_pairs(player_hand):
#     pairs = get_legal_pairs(player_hand)
#     triple_pairs = []
#     for combo in combinations(pairs, 3):
#         ranks = [card[0] // 4 for card in combo]
#         ranks.sort()
#         if ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1:
#             triple_pairs.append(combo[0] + combo[1] + combo[2])
#     return triple_pairs

# # 三同张
# def get_legal_triples(player_hand):
#     triples = []
#     for combo in combinations(player_hand, 3):
#         if combo[0] // 4 == combo[1] // 4 == combo[2] // 4:
#             triples.append(list(combo))
#     return triples

# # 三带二（夯）
# def get_legal_three_with_pair(player_hand):
#     triples = get_legal_triples(player_hand)
#     pairs = get_legal_pairs(player_hand)
#     three_with_pair = []
#     for triple in triples:
#         for pair in pairs:
#             if not set(triple) & set(pair):
#                 three_with_pair.append(triple + pair)
#     return three_with_pair

# # 顺子（五张相连单牌）
# def get_legal_straights(player_hand):
#     ranks = [card // 4 for card in player_hand]
#     straights = []
#     for combo in combinations(ranks, 5):
#         sorted_combo = sorted(combo)
#         if sorted_combo == list(range(sorted_combo[0], sorted_combo[0] + 5)):
#             straight = [card for card in player_hand if card // 4 in sorted_combo]
#             if len(straight) == 5:
#                 straights.append(straight)
#     return straights

# # 炸弹（四张或以上相同点数的牌）
# def get_legal_bombs(player_hand):
#     bombs = []
#     # 检查普通炸弹
#     for n in range(4, len(player_hand) + 1):
#         for combo in combinations(player_hand, n):
#             if all(card // 4 == combo[0] // 4 for card in combo):
#                 bombs.append(list(combo))

#     # 检查王炸（四个王）
#     joker_set = {52, 53, 106, 107}
#     if joker_set.issubset(set(player_hand)):
#         bombs.append([52, 53, 106, 107])

#     return bombs


# # 同花顺（五张相连且同花色的牌）
# def get_legal_straight_flushes(player_hand):
#     suits = {0: [], 1: [], 2: [], 3: []}
#     for card in player_hand:
#         suits[card % 4].append(card)

#     straight_flushes = []
#     for suit_cards in suits.values():
#         if len(suit_cards) >= 5:
#             ranks = [card // 4 for card in suit_cards]
#             for combo in combinations(ranks, 5):
#                 sorted_combo = sorted(combo)
#                 if sorted_combo == list(range(sorted_combo[0], sorted_combo[0] + 5)):
#                     straight_flush = [card for card in suit_cards if card // 4 in sorted_combo]
#                     if len(straight_flush) == 5:
#                         straight_flushes.append(straight_flush)
#     return straight_flushes

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


def plot_training_curves(loss_history, reward_history, weight_history, save_path="targetDQN/training_curves.png"):
    plt.figure(figsize=(15, 5))

    # Plot Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot Reward Curve
    plt.subplot(1, 3, 2)
    plt.plot(reward_history, label="Reward", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Plot Weights Change Curve
    plt.subplot(1, 3, 3)
    for i in range(len(weight_history[0])):
        plt.plot([weights[i] for weights in weight_history], label=f"Weight {i}" if i < 5 else None, alpha=0.5)
    plt.xlabel("Episodes")
    plt.ylabel("Weight Value")
    plt.title("Low-Level Network Weights")
    plt.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    plt.show()

# Training Loop
def train():
    env = GuanDanEnv()
    input_dim = 324  # 手牌 + 上次出牌 + 历史出牌记录
    high_level_action_space = ["单张", "对子", "三带", "炸弹", "过"]
    high_level_actions = 5  # 高层网络动作：单张、对子、三带、炸弹、Pass
    low_level_actions = 108  # 底层网络动作：从手牌中选择具体组合
    card_mapping = [utils.ChineseNum2Poker(i) for i in range(108)]
    loss_history = []
    reward_history = []
    weight_history = []

    agent = HierarchicalDQNAgent(input_dim, high_level_actions, low_level_actions)
    
    print(f"Validate cardmapping: {card_mapping}")
    # 加载之前的模型权重
    try:
        agent.high_level_net.load_state_dict(torch.load("targetDQN/high_level_dqn_guandan.pth", map_location=DEVICE))
        agent.low_level_net.load_state_dict(torch.load("targetDQN/low_level_dqn_guandan.pth", map_location=DEVICE))
        print("模型权重加载成功，继续训练。")
    except FileNotFoundError:
        print("未找到权重文件，将从头开始训练。")
        
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        current_player = 0
        # print(f"Episode {episode}")
        while not done:
            player_obs = obs[current_player]
            state = torch.FloatTensor(encode_obs(player_obs)).to(DEVICE)
            # Step 1: High-Level action selection (e.g., Single, Pair, Triple, Bomb, Pass)
            high_level_action = agent.select_high_level_action(state)

            # Step 2: Low-Level action selection
            if high_level_action in range(5):
                low_level_action = agent.select_low_level_action(state, player_obs['deck'], high_level_action)
            else:
                low_level_action = []

            if episode % 1000 == 0:
                print(f"Player {current_player}: {[card_mapping[i] for i in player_obs['deck']]}\n \
                      high level action: {high_level_action_space[high_level_action]}\n \
                      low level action: {[card_mapping[i] for i in low_level_action]}")
                # print(f"Selected action index: {action}, Legal actions length: {len(legal_actions)}")

            # 执行选择的动作
            response = {'player': current_player, 'action': low_level_action, 'claim': low_level_action}
            # Step后下一个状态
            next_obs = env.step(response)
            # 当前玩家出牌后的reward
            reward = env._get_obs(current_player)[current_player]['reward']
            total_reward += reward
            done = env.done
            
            # 处理游戏结束时的 next_state
            if done:
                next_state = torch.zeros_like(state)  # 或者使用 state 表示没有后续状态
            else:
                current_player = (current_player + 1) % 4
                # player_obs = obs[current_player]
                next_state = torch.FloatTensor(encode_obs(next_obs[current_player])).to(DEVICE)
                obs = next_obs
                
            # 存储经验
            # print(f"low_level_action: {low_level_action}")
            agent.memory.push(state.cpu(), (high_level_action, low_level_action), reward, next_state.cpu(), done)

            # 更新网络
            agent.update()

        # 记录奖励
        reward_history.append(total_reward)

        # 每次更新后记录低层网络的权重
        if episode % 100 == 0:
            weight_history.append(agent.low_level_net.fc1.weight.clone().detach().cpu().numpy().flatten())

        # 记录损失值
        if len(agent.memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = zip(*agent.memory.sample(BATCH_SIZE))

            # 将 low_level_actions 转换为独热编码
            low_level_actions_one_hot = []
            utils.Action2Onehot(actions, low_level_actions_one_hot)

            # 转换为 PyTorch 张量
            low_level_actions_one_hot = torch.FloatTensor(low_level_actions_one_hot).to(DEVICE)

            states = torch.FloatTensor(np.array(states)).to(DEVICE)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
            next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

            # 计算 Q 值
            q_values = (agent.low_level_net(states) * low_level_actions_one_hot).sum(dim=1, keepdim=True)
            next_q_values = agent.low_level_target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            # 计算损失
            loss = F.mse_loss(q_values, target_q_values)
            loss_history.append(loss.item())

        if episode % TARGET_UPDATE == 0:
            agent.update_target_networks()
            
        if episode % 1000  == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            print("---")
            
    # 绘制权重变化
    plot_training_curves(loss_history, reward_history, weight_history)

    torch.save(agent.high_level_net.state_dict(), "targetDQN/high_level_dqn_guandan.pth")
    torch.save(agent.low_level_net.state_dict(), "targetDQN/low_level_dqn_guandan.pth")

    # 删除不再需要的对象
    # del agent
    # gc.collect()
    # torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
