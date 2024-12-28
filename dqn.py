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
NUM_CORES = 1  # 指定使用的 CPU 核心数量
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 5000
MEMORY_SIZE = 100000
TARGET_UPDATE = 10000
DEBUG_UPDATE = 10000
EPISODES = 160000000
N_STEP = 10
EPSILON_START = 0.5
EPSILON_MIN = 0.1
RESET_INTERVAL = 1000000
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:1")
model_folder_path = "targetDQN"
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
        return self.fc3(x)  # 使用 sigmoid 确保输出在 [0, 1] 范围内

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
       
        # 初始化 epsilon 和 epsilon 衰减参数
        self.epsilon = EPSILON_START  # 初始探索概率
        self.epsilon_min = EPSILON_MIN  # 最小探索概率
        self.epsilon_decay = 0.99  # 每个 episode 的衰减率

        # 初始化目标网络权重
        self.high_level_target_net.load_state_dict(self.high_level_net.state_dict())
        self.low_level_target_net.load_state_dict(self.low_level_net.state_dict())

    def update_epsilon(self):
        """在每个 episode 结束后调用此函数来更新 epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_epsilon_periodic(self, episode):
        if episode % RESET_INTERVAL == 0:
            return EPSILON_START
        decay_rate = (EPSILON_START - EPSILON_MIN) / RESET_INTERVAL
        self.epsilon = max(EPSILON_MIN, EPSILON_START - decay_rate * (episode % RESET_INTERVAL))
        
    def select_high_level_action(self, state):
        state = state.to(DEVICE)
        if random.random() < self.epsilon:
            return random.randint(0, self.high_level_actions - 1)
        else:
            with torch.no_grad():
                action = self.high_level_net(state)
                print(f"action: {action}")
                return action.argmax().item()

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
            
        print(f"q_values: {q_values}")

        # 筛选当前手牌中的牌，并根据 Q 值排序
        # hand_q_values = [(card, q_values[card]) for card in hand]
        # hand_q_values.sort(key=lambda x: x[1], reverse=True)

        # 根据高层决策选择相应的牌型
        if high_level_action == 0:  # 单张
            possible_combinations = [[card] for card in hand]  # 保证输出是列表
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

        # 如果没有找到符合条件的组合，则返回 Pass
        if not possible_combinations:
            return []
        # else: 
        #     print(f"Possible combo: {possible_combinations}")
    
        # 根据 Q 值对可能的组合排序
        combination_q_values = [(combo, sum(q_values[card] for card in combo)) for combo in possible_combinations]
        combination_q_values.sort(key=lambda x: x[1], reverse=True)

        # 返回 Q 值最高的组合
        return combination_q_values[0][0]

    def update_target_networks(self):
        self.high_level_target_net.load_state_dict(self.high_level_net.state_dict())
        self.low_level_target_net.load_state_dict(self.low_level_net.state_dict())

    def update(self, global_memory, beta=0.4, n_step=N_STEP):
        if len(global_memory) < BATCH_SIZE:
            return 0, 0, 0, 0

        # 从记忆库中采样
        # batch = global_memory.sample(BATCH_SIZE)
        batch, indices, weights = global_memory.sample(BATCH_SIZE, beta)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        # 解包高层和低层动作和奖励
        high_level_actions, low_level_actions = zip(*actions)   
        high_level_rewards, low_level_rewards = zip(*rewards)   
           
        # 低层动作转换为独热编码
        low_level_actions_one_hot = []
        utils.Action2Onehot(low_level_actions, low_level_actions_one_hot)

        # 转换为 PyTorch 张量
        low_level_actions_one_hot = torch.FloatTensor(np.array(low_level_actions_one_hot)).to(DEVICE)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        high_level_actions = torch.LongTensor(high_level_actions).unsqueeze(1).to(DEVICE)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        weights = torch.FloatTensor(np.array(weights)).to(DEVICE)

        # 计算 n-step 累积奖励
        discounted_low_level_rewards, discounted_high_level_rewards = [], []
        for i in range(len(low_level_rewards)):
            low_G, high_G = 0, 0
            for j in range(n_step):
                if i + j < len(low_level_rewards):
                    low_G += (GAMMA ** j) * low_level_rewards[i + j]  # 累计低层奖励
                    high_G += (GAMMA ** (j * 2)) * high_level_rewards[i + j]  # 累计高层奖励（较慢衰减）
                if i + j < len(dones) and dones[i + j]:
                    break
            discounted_low_level_rewards.append(low_G)
            discounted_high_level_rewards.append(high_G)

        # 转换为 PyTorch 张量
        discounted_low_level_rewards = torch.FloatTensor(discounted_low_level_rewards).unsqueeze(1).to(DEVICE)
        discounted_high_level_rewards = torch.FloatTensor(discounted_high_level_rewards).unsqueeze(1).to(DEVICE)
        
        # 更新低层网络
        low_level_loss, low_level_q_value = self._update_low_level_network(states, low_level_actions_one_hot, discounted_low_level_rewards, next_states, dones, indices, weights)

        # 更新高层网络
        high_level_loss, high_level_q_value = self._update_high_level_network(states, high_level_actions, discounted_high_level_rewards, next_states, dones, indices, weights)
        
        return low_level_loss, low_level_q_value, high_level_loss, high_level_q_value
    
    def _update_low_level_network(self, states, actions_one_hot, rewards, next_states, dones, indices, weights, n_step=N_STEP):
        q_values = (self.low_level_net(states) * actions_one_hot).sum(dim=1, keepdim=True)
        next_q_values = self.low_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA ** n_step) * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        loss = (F.mse_loss(q_values, target_q_values, reduction='none') * weights).mean()

        self.low_level_optimizer.zero_grad()
        loss.backward()
        self.low_level_optimizer.step()
        
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        global_memory.update_priorities(indices, td_errors)
    
        return loss.item(), q_values.mean().item()

    def _update_high_level_network(self, states, actions, rewards, next_states, dones, indices, weights, n_step=N_STEP):
        q_values = self.high_level_net(states).gather(1, actions)
        next_q_values = self.high_level_target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA ** n_step) * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        loss = (F.mse_loss(q_values, target_q_values, reduction='none') * weights).mean()

        self.low_level_optimizer.zero_grad()
        loss.backward()
        self.high_level_optimizer.step()
        
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        global_memory.update_priorities(indices, td_errors)
    
        return loss.item(), q_values.mean().item()

    
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
        high_level_actions = 11  # 高层网络动作：单张、对子、三带、炸弹、Pass...
        low_level_actions = 108  # 底层网络动作：从手牌中选择具体组合
        card_mapping = [utils.ChineseNum2Poker(i) for i in range(108)]
        agents = [HierarchicalDQNAgent(input_dim, high_level_actions, low_level_actions) for _ in range(NUM_AGENTS)]
        
        # 初始化指标记录
        agent_metrics = [{'low_level_loss': [], 'high_level_loss': [], 'reward': [], 'low_level_q_value': [], 'high_level_q_value': []} for _ in range(NUM_AGENTS)]
        
        high_level_action_space = ["单张", "对子", "三带", "炸弹", "三连对（木板）", "三同连张（钢板）", "三带二（夯）", "顺子", "同花顺", "火箭（王炸）", "过"]
        high_level_action_picks = [ 0 for i in range(high_level_actions)]
        wins = 0
      
        # print(f"Validate cardmapping: {card_mapping}")
        # 加载之前的模型权重
        try:
            # agent.high_level_net.load_state_dict(torch.load("targetDQN/high_level_dqn_guandan.pth", map_location=DEVICE))
            # agent.low_level_net.load_state_dict(torch.load("targetDQN/low_level_dqn_guandan.pth", map_location=DEVICE))
            for i, agent in enumerate(agents):
                agent.high_level_net.load_state_dict(torch.load(f"{model_folder_path}/agent_{i}_high_level.pth", map_location=DEVICE))
                agent.low_level_net.load_state_dict(torch.load(f"{model_folder_path}/agent_{i}_low_level.pth", map_location=DEVICE))
            print("模型权重加载成功，继续训练。")
        except FileNotFoundError:
            print("未找到权重文件，将从头开始训练。")
            
        episode = 0
        # for episode in range(EPISODES // mp.cpu_count()):
        while True:
            obs = env.reset()
            episode += 1
            # done = False
            total_reward = 0
            current_player = 0
            # episode_rewards = [0] * NUM_AGENTS  # 每个智能体的回合总奖励
            # print(f"Episode {episode}")
            while not env.done:
                agent = agents[current_player]
                player_obs = obs[current_player]
                high_level_reward = 0
                low_level_reward = 0
                # 如果当前玩家出完牌了
                if current_player in env.cleared:
                    current_player = (current_player + 1) % 4
                    continue
                
                state = torch.FloatTensor(encode_obs(player_obs)).to(DEVICE)                
                
                # Step 1: High-Level action selection (e.g., Single, Pair, Triple, Bomb, Pass)
                high_level_action = agent.select_high_level_action(state)
                
                # 记录动作选取次数
                high_level_action_picks[high_level_action] += 1
                
                # Step 2: Low-Level action selection
                # 出牌
                if high_level_action in range(high_level_actions - 1):
                    low_level_action = agent.select_low_level_action(state, player_obs['deck'], high_level_action)
                    if low_level_action:
                        high_level_reward = 3
                    else:
                        high_level_reward = -10
                # 过
                else:
                    high_level_reward = -1
                    low_level_reward = 0
                    low_level_action = []

                # 执行选择的动作
                response = {'player': current_player, 'action': low_level_action, 'claim': low_level_action}
                # Step后下一个状态
                next_obs = env.step(response)
                obs = next_obs
                
                # 开始计算奖励
                if env.game_state_info == f"Player {current_player}: ILLEGAL PASS AS FIRST-HAND":
                    high_level_reward = -10
                elif env.game_state_info == f"Player {current_player}: POKERTYPE MISMATCH":
                    high_level_reward = -5
                elif env.game_state_info == f"Player {current_player}: CANNOT BEAT LASTMOVE":
                    high_level_reward = -3
                    low_level_reward = -5
                elif low_level_action: 
                    low_level_reward = 3 # env._get_obs(current_player)[current_player]['reward']
                
                # Debug信息
                if episode % DEBUG_UPDATE == 0:
                    print(f"Player {current_player}: {[card_mapping[i] for i in player_obs['deck']]}")
                    print(f"high level action: {high_level_action_space[high_level_action]}")
                    print(f"low level action: {[card_mapping[i] for i in low_level_action]}")
                    print(f"获得了 high level reward: {high_level_reward}, low level reward: {low_level_reward}")

                # 记录画图数据
                if episode % 1000 == 0:
                    low_level_loss, low_level_q_value, high_level_loss, high_level_q_value = agent.update(global_memory)
                    agent_metrics[current_player]['low_level_loss'].append(low_level_loss)
                    agent_metrics[current_player]['low_level_q_value'].append(low_level_q_value)
                    agent_metrics[current_player]['high_level_loss'].append(high_level_loss)
                    agent_metrics[current_player]['high_level_q_value'].append(high_level_q_value)
                    
                    # agent_metrics[i]['reward'].append(episode_rewards[i])

                # 处理游戏结束时的 next_state
                if env.done:
                    if env.game_state_info == "Finished":
                        print("A successful end game!")
                        wins += 1
                        # 获取每个玩家的最终奖励
                        print(f"Final rewards: {obs[player_id][reward]}")

                        # 将最终奖励存储到 global_memory 中
                        for player_id in range(NUM_AGENTS):
                            # 假设 state 和 next_state 在最后一轮已经更新
                            final_state = torch.FloatTensor(encode_obs(obs[player_id])).to(DEVICE)
                            global_memory.push(
                                final_state.cpu(), 
                                None,  # 最后一轮没有动作
                                (obs[player_id][reward], 0),  # 只记录高层奖励
                                torch.zeros_like(final_state).cpu(),  # 游戏结束时的 next_state 为零向量
                                True  # 标志游戏结束
                            )
                            break
                    elif episode % DEBUG_UPDATE == 0:
                        print(f"A faulty game by {env.game_state_info}")
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
                global_memory.push(state.cpu(), (high_level_action, low_level_action), (high_level_reward, low_level_reward), next_state.cpu(), env.done)
                                       
            # 在 episode 结束后更新 epsilon         
            if episode % TARGET_UPDATE == 0:
                for agent in agents:
                    agent.update_target_networks()
            agent.update_epsilon_periodic(episode)
                
            if episode % DEBUG_UPDATE == 0:
                # print(f"Episode {episode}, Total Reward: {total_reward}")
                print(f"Episode {episode}, 当前 Epsilon: {agents[0].epsilon}")
                print(f"总计出现了 {', '.join([f'{high_level_action_space[i]}: {high_level_action_picks[i]}' for i in range(high_level_actions)])}")
                print("---")
                utils.plot_agent_metrics(agent_metrics, model_folder_path)

                        
            # 保存权重
            if episode % 100000 == 0:
                for i, agent in enumerate(agents):
                    torch.save(agent.high_level_net.state_dict(), f"{model_folder_path}/agent_{i}_high_level.pth")
                    torch.save(agent.low_level_net.state_dict(), f"{model_folder_path}/agent_{i}_low_level.pth")
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
        
    

