import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


class Utils():
    
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.jokers = ['jo', 'jO']
        
        self.chineseCardscale = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        self.chineseSuitset = ['红心','方块','黑桃','草花']
    
    def Num2Poker(self, num: int):
        num_in_deck = num % 54
        if num_in_deck == 52:
            return "jo"
        if num_in_deck == 53:
            return "jO"
        # Normal cards:
        pokernumber = self.cardscale[num_in_deck // 4]
        pokersuit = self.suitset[num_in_deck % 4]
        return pokersuit + pokernumber
    
    def ChineseNum2Poker(self, num: int):
        num_in_deck = num % 54
        if num_in_deck == 52:
            return "小王"
        if num_in_deck == 53:
            return "大王"
        # Normal cards:
        pokernumber = self.chineseCardscale[num_in_deck // 4]
        pokersuit = self.chineseSuitset[num_in_deck % 4]
        return pokersuit + pokernumber
    
    def Poker2Num(self, poker: str, deck):
        num_in_deck = -1
        if poker[1] == "o":
            num_in_deck = 52
        elif poker[1] == "O":
            num_in_deck = 53
        else:
            num_in_deck = self.cardscale.index(poker[1])*4 + self.suitset.index(poker[0])
        if num_in_deck == -1:
            return -1
        if num_in_deck in deck:
            return num_in_deck
        return num_in_deck + 54
    
    def Action2Onehot(self, low_level_actions, low_level_actions_one_hot):
        for action in low_level_actions:
            one_hot = np.zeros(108, dtype=np.float32)
            if action and action[0] != -1:
                for card in action:
                    one_hot[card] = 1
            low_level_actions_one_hot.append(one_hot)
            
    # 单张
    def get_legal_singles(self, hand):
        # return [[card] for card in hand]
        return [hand[0]]

    def get_legal_pairs(self, hand, callFromSelf):
        pairs = []
        jokers = [52, 53, 106, 107]

        # 查找普通对子
        for combo in combinations(hand, 2):
            if combo[0] // 4 == combo[1] // 4:
                if not callFromSelf:
                    return list(combo)
                pairs.append(list(combo))

        # 特殊处理两个小王或两个大王作为对子
        small_jokers = [j for j in hand if j in [52, 106]]
        if len(small_jokers) >= 2:
            if not callFromSelf:
                return [small_jokers[0], small_jokers[1]]
            pairs.append([small_jokers[0], small_jokers[1]])

        big_jokers = [j for j in hand if j in [53, 107]]
        if len(big_jokers) >= 2:
            if not callFromSelf:
                return [big_jokers[0], big_jokers[1]]
            pairs.append([big_jokers[0], big_jokers[1]])

        return pairs

    # 三同张
    def get_legal_triples(self, hand, callFromSelf):
        triples = []
        for combo in combinations(hand, 3):
            # 确保三张牌的点数相同，并且不包含大小王
            if (combo[0] // 4 == combo[1] // 4 == combo[2] // 4 and
                all(card not in [52, 53, 106, 107] for card in combo)):
                if not callFromSelf:
                    return list(combo)
                triples.append(list(combo))
        return triples

    # 炸弹（四张或以上相同点数的牌）
    def get_legal_bombs(self, hand):
        bombs = []

        # 统计每个点数的出现次数（不包括大小王）
        rank_counts = Counter(card // 4 for card in hand if card not in [52, 53, 106, 107])

        # 查找四张或以上相同点数的牌
        for rank, count in rank_counts.items():
            if count >= 4:
                bomb = [card for card in hand if card // 4 == rank]
                # bombs.append(bomb[:count])  # 取前 count 张牌
                return bomb[:count]

        return bombs

    def get_legal_pairs_without_jokers(self, hand):
        # 获取普通对子，不包含大小王
        jokers = [52, 53, 106, 107]
        pairs = []
        
        for combo in combinations(hand, 2):
            if combo[0] // 4 == combo[1] // 4 and combo[0] not in jokers and combo[1] not in jokers:
                pairs.append(list(combo))
        
        return pairs

    # 三连对（木板）
    def get_legal_triple_pairs(self, hand):
        pairs = self.get_legal_pairs_without_jokers(hand)
        triple_pairs = []
        for combo in combinations(pairs, 3):
            ranks = [pair[0] // 4 for pair in combo]
            ranks.sort()
            if ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1:
                # triple_pairs.append(combo[0] + combo[1] + combo[2])
                return combo[0] + combo[1] + combo[2]
        return triple_pairs

    # 三带二（夯）
    def get_legal_three_with_pair(self, hand):
        triples = self.get_legal_triples(hand, True)
        pairs = self.get_legal_pairs(hand, True)
        three_with_pair = []
        for triple in triples:
            for pair in pairs:
                if not set(triple) & set(pair):
                    # three_with_pair.append(triple + pair)
                    return triple + pair
        return three_with_pair

    # 顺子（五张相连单牌）
    def get_legal_straights(self, hand):
        straights = []
        # 过滤掉大小王
        filtered_hand = [card for card in hand if card not in [52, 53, 106, 107]]
        ranks = [card // 4 for card in filtered_hand]

        for combo in combinations(ranks, 5):
            sorted_combo = sorted(combo)
            if sorted_combo == list(range(sorted_combo[0], sorted_combo[0] + 5)):
                straight = [card for card in filtered_hand if card // 4 in sorted_combo]
                if len(straight) == 5:
                    # straights.append(straight)
                    return straight
        return straights

    # 同花顺（五张相连且同花色的牌）
    def get_legal_straight_flushes(self, hand):
        straight_flushes = []
        suits = {0: [], 1: [], 2: [], 3: []}

        # 过滤掉大小王
        filtered_hand = [card for card in hand if card not in [52, 53, 106, 107]]
        for card in filtered_hand:
            suits[card % 4].append(card)

        for suit_cards in suits.values():
            if len(suit_cards) >= 5:
                ranks = [card // 4 for card in suit_cards]
                for combo in combinations(ranks, 5):
                    sorted_combo = sorted(combo)
                    if sorted_combo == list(range(sorted_combo[0], sorted_combo[0] + 5)):
                        straight_flush = [card for card in suit_cards if card // 4 in sorted_combo]
                        if len(straight_flush) == 5:
                            # straight_flushes.append(straight_flush)
                            return straight_flush
        return straight_flushes

    # 三同连张（钢板）
    def get_legal_triple_straight(self, hand):
        triples = self.get_legal_triples(hand, True)
        triple_straights = []
        for combo in combinations(triples, 2):
            ranks = [triple[0] // 4 for triple in combo]
            if ranks[1] == ranks[0] + 1:
                # triple_straights.append(combo[0] + combo[1])
                return combo[0] + combo[1]
        return triple_straights

    # 火箭（王炸）
    def get_legal_rockets(self, hand):
        if set([52, 53, 106, 107]).issubset(hand):
            return [52, 53, 106, 107]
        return []
            
    def plot_agent_metrics(self, agent_metrics, save_path):
        for i, metrics in enumerate(agent_metrics):
            plt.figure(figsize=(15, 10))

            # 高层 Loss
            plt.subplot(2, 2, 1)
            plt.plot(metrics['high_level_loss'], label=f'Agent {i} High-Level Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title(f'Agent {i} High-Level Loss')
            plt.legend()

            # 低层 Loss
            plt.subplot(2, 2, 2)
            plt.plot(metrics['low_level_loss'], label=f'Agent {i} Low-Level Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title(f'Agent {i} Low-Level Loss')
            plt.legend()

            # 高层 Q-Value
            plt.subplot(2, 2, 3)
            plt.plot(metrics['high_level_q_value'], label=f'Agent {i} High-Level Q-Value')
            plt.xlabel('Steps')
            plt.ylabel('Q-Value')
            plt.title(f'Agent {i} High-Level Q-Value')
            plt.legend()

            # 低层 Q-Value
            plt.subplot(2, 2, 4)
            plt.plot(metrics['low_level_q_value'], label=f'Agent {i} Low-Level Q-Value')
            plt.xlabel('Steps')
            plt.ylabel('Q-Value')
            plt.title(f'Agent {i} Low-Level Q-Value')
            plt.legend()

            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
    
class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
    
    def __str__(self):
        return self.ErrorInfo  
    