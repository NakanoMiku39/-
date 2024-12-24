import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from env import GuanDanEnv
from itertools import combinations
import matplotlib.pyplot as plt
import gc
from utils import Utils
from ReplayMemory import ReplayMemory
import torch.multiprocessing as mp
import traceback

# Hyperparameters
NUM_AGENTS = 4
NUM_CORES = 20  # 指定使用的 CPU 核心数量
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 5000
MEMORY_SIZE = 100000
TARGET_UPDATE = 10000
DEBUG_UPDATE = 1000
EPISODES = 400000
CLIP_EPSILON = 0.2  # 默认值
ENTROPY_COEFF = 0.01  # 控制熵奖励权重
RESET_INTERVAL = 100000
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0")
model_folder_path = "PPO"
utils = Utils()
# DEVICE = torch.device("cpu")

print(f"Using {DEVICE}")
print(torch.cuda.is_available())  # 如果返回 False，表示 GPU 不可用
print(torch.cuda.device_count())  # 查看可用的 GPU 数量
print(torch.cuda.current_device())  # 获取当前 GPU 设备编号
print(torch.cuda.get_device_name(0))  # 查看 GPU 的名称（如果存在）
print(torch.__version__)
print(torch.version.cuda)

global_memory = ReplayMemory(MEMORY_SIZE)  # 全局共享经验池

# DQN Network
class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        action_logits = torch.clamp(action_logits, min=-10, max=10)  # 裁剪 logits
        state_value = self.value_head(x)
        return F.softmax(action_logits, dim=-1), state_value
    
class HierarchicalPPOAgent:
    def __init__(self, input_dim, high_level_actions, low_level_actions):
        self.high_level_net = PPOPolicyNetwork(input_dim, high_level_actions).to(DEVICE)
        self.low_level_net = PPOPolicyNetwork(input_dim, low_level_actions).to(DEVICE)

        self.high_level_optimizer = optim.Adam(self.high_level_net.parameters(), lr=LR)
        self.low_level_optimizer = optim.Adam(self.low_level_net.parameters(), lr=LR)
               
    def select_high_level_action(self, state):
        state = state.to(DEVICE)
        with torch.no_grad():
            action_probs, _ = self.high_level_net(state)

            # 确保 action_probs 是有效的概率分布
            action_probs = torch.softmax(action_probs, dim=-1)
            if torch.any(torch.isnan(action_probs)):
                print(f"Invalid action_probs detected: {action_probs}")
                action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]  # 平均分布兜底策略

            # 创建分布并采样
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            return action.item(), action_dist.log_prob(action)

    def select_low_level_action(self, state, hand, high_level_action):
        state = state.to(DEVICE)
        with torch.no_grad():
            action_probs, _ = self.low_level_net(state)
            q_values = action_probs.cpu().numpy()

        if high_level_action == 0:  # 单张
            possible_combinations = [[card] for card in hand]
        elif high_level_action == 1:  # 对子
            possible_combinations = utils.get_legal_pairs(hand)
        elif high_level_action == 2:  # 三同张
            possible_combinations = utils.get_legal_triples(hand, False)
        elif high_level_action == 3:  # 炸弹
            possible_combinations = utils.get_legal_bombs(hand)
        elif high_level_action == 4:  # 三连对（木板）
            possible_combinations = utils.get_legal_triple_pairs(hand)
        elif high_level_action == 5:  # 三同连张（钢板）
            possible_combinations = utils.get_legal_triple_straight(hand)
        elif high_level_action == 6:  # 三带二（夯）
            possible_combinations = utils.get_legal_three_with_pair(hand)
        elif high_level_action == 7:  # 顺子
            possible_combinations = utils.get_legal_straights(hand)
        elif high_level_action == 8:  # 同花顺
            possible_combinations = utils.get_legal_straight_flushes(hand)
        elif high_level_action == 9:  # 火箭（王炸）
            possible_combinations = utils.get_legal_rockets(hand)

        if not possible_combinations:
            return [], torch.tensor(0.0)

        combination_q_values = [(combo, sum(q_values[card] for card in combo)) for combo in possible_combinations]
        combination_q_values.sort(key=lambda x: x[1], reverse=True)
        best_combo = combination_q_values[0][0]
        best_value = torch.tensor(combination_q_values[0][1])

        return best_combo, best_value

    def update_policy(self, policy_net, optimizer, states, actions, rewards, old_log_probs, advantages, is_low_level=False):
        # 前向传播
        action_probs, state_values = policy_net(states)

        if is_low_level:
            # 对于低层独热编码，直接计算每个动作的概率
            log_probs = torch.sum(actions * torch.log(action_probs + 1e-10), dim=1, keepdim=True)
        else:
            # 对于高层动作，确保 actions 是二维张量
            actions = actions.long().unsqueeze(1)
            log_probs = torch.log(action_probs.gather(1, actions))

        # 计算比率和损失
        ratios = torch.exp(log_probs - old_log_probs)
        clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        value_loss = F.mse_loss(state_values.squeeze(), rewards)
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()

        # 总损失
        loss = policy_loss + 0.5 * value_loss - ENTROPY_COEFF * entropy_loss

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update(self, memory):
        if len(memory) < BATCH_SIZE:
            return

        # batch = memory.sample(BATCH_SIZE)

        # 解包每一条经验
        batch, indices, weights = memory.sample(BATCH_SIZE)
        states, actions, rewards, old_log_probs, advantages = zip(*batch)

        # 转换为张量
        states = torch.stack(states).to(DEVICE)
        
        high_actions, low_actions = zip(*actions)
        high_actions = torch.tensor(high_actions).to(DEVICE)
        
        low_level_actions_one_hot = []
        utils.Action2Onehot(low_actions, low_level_actions_one_hot)
        low_level_actions_one_hot = torch.FloatTensor(np.array(low_level_actions_one_hot)).to(DEVICE)
        
        high_rewards, low_rewards = zip(*rewards)
        high_rewards = torch.tensor(high_rewards, dtype=torch.float).to(DEVICE)
        low_rewards = torch.tensor(low_rewards, dtype=torch.float).to(DEVICE)
        
        high_log_probs, low_log_probs = zip(*old_log_probs)
        # 确保 log_probs 是一维张量
        high_log_probs = torch.stack([log_prob.unsqueeze(0) for log_prob in high_log_probs]).to(DEVICE)
        low_log_probs = torch.stack([log_prob.unsqueeze(0) for log_prob in low_log_probs]).to(DEVICE)
        
        advantages = torch.stack(advantages).to(DEVICE)

        # 更新高层和低层网络
        self.update_policy(self.high_level_net, self.high_level_optimizer, states, high_actions, high_rewards, high_log_probs, advantages, is_low_level=False,)
        self.update_policy(self.low_level_net, self.low_level_optimizer, states, low_level_actions_one_hot, low_rewards, low_log_probs, advantages, is_low_level=True,)

    
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
def train(rank):
    try:
        print(f"Process {rank} started")

        env = GuanDanEnv()
        input_dim = 324  # 手牌 + 上次出牌 + 历史出牌记录
        high_level_actions = 11
        low_level_actions = 108

        agents = [HierarchicalPPOAgent(input_dim, high_level_actions, low_level_actions) for _ in range(NUM_AGENTS)]
        
        agent_metrics = [{'low_level_loss': [], 'high_level_loss': [], 'reward': [], 'low_level_q_value': [], 'high_level_q_value': []} for _ in range(NUM_AGENTS)]
        card_mapping = [utils.ChineseNum2Poker(i) for i in range(108)]
        high_level_action_space = ["单张", "对子", "三带", "炸弹", "三连对（木板）", "三同连张（钢板）", "三带二（夯）", "顺子", "同花顺", "火箭（王炸）", "过"]
        high_level_action_picks = [ 0 for i in range(high_level_actions)]
        wins = 0
        
        try:
            for i, agent in enumerate(agents):
                agent.high_level_net.load_state_dict(torch.load(f"{model_folder_path}/agent_{i}_high_level.pth", map_location=DEVICE))
                agent.low_level_net.load_state_dict(torch.load(f"{model_folder_path}/agent_{i}_low_level.pth", map_location=DEVICE))
            print("模型权重加载成功，继续训练。")
        except FileNotFoundError:
            print("未找到权重文件，将从头开始训练。")
            
        
        for episode in range(EPISODES):
            obs = env.reset()
            current_player = 0
            done = False
            agent_metrics = [{'low_level_loss': [], 'high_level_loss': [], 'reward': [], 'low_level_q_value': [], 'high_level_q_value': []} for _ in range(NUM_AGENTS)]

            while not done:
                # 如果当前玩家出完牌了
                if current_player in env.cleared:
                    current_player = (current_player + 1) % 4
                    continue
                
                player_obs = obs[current_player]
                state = torch.FloatTensor(encode_obs(player_obs)).to(DEVICE)
                agent = agents[current_player]
                high_reward = 0
                low_reward = 0

                # High-level action selection
                high_action, high_log_prob = agent.select_high_level_action(state)
                # 记录动作选取次数
                high_level_action_picks[high_action] += 1

                # Low-level action selection
                if high_action in range(high_level_actions - 1):
                    low_action, low_log_prob = agent.select_low_level_action(state, player_obs['deck'], high_action)
                    if low_action:
                        high_reward = 3
                    else:
                        high_reward = -10
                # 过
                else:
                    high_reward = -1
                    low_reward = 0
                    low_action = []
                    low_log_prob = torch.tensor(0.0)  # 空动作时对数概率为零

                # Execute action
                response = {'player': current_player, 'action': low_action, 'claim': low_action}
                next_obs = env.step(response)

                obs = next_obs
                done = env.done
                
                if env.game_state_info == f"Player {current_player}: ILLEGAL PASS AS FIRST-HAND":
                    high_reward = -10
                elif env.game_state_info == f"Player {current_player}: POKERTYPE MISMATCH":
                    high_reward = -5
                elif env.game_state_info == f"Player {current_player}: CANNOT BEAT LASTMOVE":
                    high_reward = -3
                    low_reward = -5
                elif low_action: 
                    low_reward = 3
                
                # Debug信息
                if episode % DEBUG_UPDATE == 0:
                    print(f"Player {current_player}: {[card_mapping[i] for i in player_obs['deck']]}")
                    print(f"high level action: {high_level_action_space[high_action]}")
                    print(f"low level action: {[card_mapping[i] for i in low_action]}")
                    print(f"获得了 high level reward: {high_reward}, low level reward: {low_reward}")
                    
                if env.done:
                    if env.game_state_info == "Finished":
                        print("A successful end game!")
                        wins += 1
                        # 获取每个玩家的最终奖励
                        for player_id in range(NUM_AGENTS):
                            print(f"Final rewards for Player {player_id}: {obs[player_id]['reward']}")
                        break
                    elif episode % DEBUG_UPDATE == 0:
                        print(f"A faulty game by {env.game_state_info}")
                    next_state = torch.zeros_like(state)  # 游戏结束时，设置零向量
                else:
                    # 下一个玩家
                    current_player = (current_player + 1) % NUM_AGENTS
                    while current_player in env.cleared:
                        current_player = (current_player + 1) % NUM_AGENTS
                    next_state = torch.FloatTensor(encode_obs(next_obs[current_player])).to(DEVICE)
                    
                # 计算奖励和优势
                _, current_value = agent.high_level_net(state)
                _, next_value = agent.high_level_net(next_state)
                delta = high_reward + GAMMA * next_value * (1 - done) - current_value
                advantage = delta.detach()

                
                # Store experience in global_memory
                global_memory.push(
                    state.cpu(),  # 当前状态
                    (high_action, low_action),  # 动作
                    (high_reward, low_reward),  # 奖励
                    (high_log_prob.cpu(), low_log_prob.cpu()),  # 动作对数概率
                    advantage.cpu(),  # 优势
                )

            # Update agent policies
            for agent in agents:
                agent.update(global_memory)

            if episode % TARGET_UPDATE == 0:
                for i, agent in enumerate(agents):
                    torch.save(agent.high_level_net.state_dict(), f"{model_folder_path}/agent_{i}_high_level.pth")
                    torch.save(agent.low_level_net.state_dict(), f"{model_folder_path}/agent_{i}_low_level.pth")
                    print(f"Model {i} Saved")

            if episode % DEBUG_UPDATE == 0:
                # print(f"Episode {episode}, Total Reward: {total_reward}")
                print(f"Episode {episode}")
                print(f"总计出现了 {', '.join([f'{high_level_action_space[i]}: {high_level_action_picks[i]}' for i in range(high_level_actions)])}")
                print("---")
        print(f"Results:\nTotal episodes: {EPISODES}\nSuccessful end game: {wins}")
        

    except Exception as e:
        print(f"Error in process {rank}: {e}")
        print("Stack trace:")
        traceback.print_exc()
        
if __name__ == "__main__":
    mp.set_start_method('spawn')  # 设置启动方法
    processes = []
    for rank in range(NUM_CORES):
        p = mp.Process(target=train, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("Done")
    

