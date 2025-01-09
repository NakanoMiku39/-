from collections import deque
import numpy as np
import random
from multiprocessing import Manager, Queue, set_start_method
import threading

# Replay Memory
class ReplayMemory:
    # def __init__(self, capacity):
    #     self.memory = deque(maxlen=capacity)
    
    # def push(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))
    
    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def pushDQN(self, state, action, reward, next_state, done):
        max_priority = float(max(self.priorities, default=1.0))
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
    def push(self, state, action, reward, next_state, done, log_prob, advantage):
        max_priority = float(max(self.priorities, default=1.0))
        self.memory.append((state, action, reward, next_state, done, log_prob, advantage))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return []

        # 确保 priorities 是浮点数数组
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # 计算重要性采样权重
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = float(abs(td_error) + 1e-5)
            
class SharedReplayMemory:
    def __init__(self, memory, priorities, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = memory    # 使用 Manager 的 list 共享内存
        self.priorities = priorities   # 使用 Manager 的 list 共享优先级
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done, log_prob=None, advantage=None):
        """
        添加新的经验到共享经验池中。
        """
        max_priority = float(max(self.priorities, default=1.0))
        experience = (state, action, reward, next_state, done, log_prob, advantage)

        # 确保经验池不会超过容量
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # 移除最旧的经验
            self.priorities.pop(0)  # 移除对应的优先级

        self.memory.append(experience)
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        """
        从共享经验池中采样。
        """
        if len(self.memory) == 0:
            return [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # 计算重要性采样权重
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        根据 TD-Error 更新优先级。
        """
        for i, td_error in zip(indices, td_errors):
            if 0 <= i < len(self.priorities):
                self.priorities[i] = float(abs(td_error) + 1e-5)

    def __len__(self):
        return len(self.memory)

class QueueReplayMemory:
    def __init__(self, max_size):
        self.queue = Queue(maxsize=max_size)

    def push(self, state, action, reward, next_state, done, log_prob, advantage):
        if self.queue.full():
            self.queue.get()  # 丢弃最旧的经验
        self.queue.put((state, action, reward, next_state, done, log_prob, advantage))
    
    def sample(self, batch_size):
        samples = []
        for _ in range(batch_size):
            if not self.queue.empty():
                samples.append(self.queue.get())
        if len(samples) == 0:
            return None
        states, actions, rewards, next_states, dones, log_prob, advantage = zip(*samples)
        return states, actions, rewards, next_states, dones, log_prob, advantage

    def __len__(self):
        return self.queue.qsize()