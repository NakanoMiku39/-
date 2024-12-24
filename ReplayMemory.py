from collections import deque
import numpy as np
import random

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

    def push(self, state, action, reward, next_state, done):
        max_priority = float(max(self.priorities, default=1.0))
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
    # def push(self, state, action, reward, next_state, done, log_prob, advantage):
    #     max_priority = float(max(self.priorities, default=1.0))
    #     self.memory.append((state, action, reward, next_state, done, log_prob, advantage))
    #     self.priorities.append(max_priority)

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