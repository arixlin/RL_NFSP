# -*- coding: utf-8 -*-
import DQN_DouDiZhu as DQN
from collections import deque
import AveragePolicyNetwork as SLN

class RunAgent:
    """class for an agent"""
    def __init__(self, agent, player):
        self.Agent = agent
        self.player = player
        self.ACTION_NUM = agent.dim_actions
        self.STATE_NUM = agent.dim_states
        self.RLMemory_num = 200
        self.SLMemory_num = 200
        self.RLMemory = deque(maxlen=self.RLMemory_num)
        self.SLMemory = deque(maxlen=self.SLMemory_num)
        self.Q = DQN.DQN_DouDiZhu(self.ACTION_NUM, self.STATE_NUM, self.RLMemory, self.RLMemory_num, self.player)
        self.Pi = SLN.Pi(self.ACTION_NUM, self.STATE_NUM, self.SLMemory, self.SLMemory_num, self.player)
        self.ETA = 0.1
        self.EPISODE_NUM = 5000000
        self.Q_enable = False
