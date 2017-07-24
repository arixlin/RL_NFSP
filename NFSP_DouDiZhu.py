import DQN_DouDiZhu as DQN
from collections import deque
import AveragePolicyNetwork as SLN
import agent
import numpy as np
import random


class RunAgent:
    """class for an agent"""
    def __init__(self, agent):
        self.Agent = agent
        self.ACTION_NUM = agent.dim_actions
        self.STATE_NUM = agent.dim_states
        self.RLMemory_num = 1000
        self.SLMemory_num = 1000
        self.RLMemory = deque(maxlen=self.RLMemory_num)
        self.SLMemory = deque(maxlen=self.SLMemory_num)
        self.Q = DQN.DQN_DouDiZhu(self.ACTION_NUM, self.STATE_NUM, self.RLMemory)
        self.Pi = SLN.Pi(self.ACTION_NUM, self.STATE_NUM, self.SLMemory)
        self.EPSILON = 0.06
        self.ETA = 0.1
        self.EPISODE_NUM = 10000


if __name__ == '__main__':
    agent = agent.Agent(models=["rl", "rl", "rl"])
    runAgent1 = RunAgent(agent)
    runAgent2 = RunAgent(agent)
    runAgent3 = RunAgent(agent)
    for i in range(runAgent1.EPISODE_NUM):
        agent.reset()
        done = False
        while(True):
            s, actions = agent.get_actions_space(player=1)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if random.random() < runAgent1.ETA:
                action_id, label = runAgent1.Q.getAction(actions_ont_hot, s)
                if label:
                    runAgent1.SLMemory.append([s, action_id])
            else:
                action_id = runAgent1.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            action_id = actions.index(action_id)
            done = agent.step(player=1, action_id=action_id)
            if done:
                break

            s, actions = agent.get_actions_space(player=2)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if random.random() < runAgent2.ETA:
                action_id, label = runAgent2.Q.getAction(actions_ont_hot, s)
                if label:
                    runAgent2.SLMemory.append([s, action_id])
            else:
                action_id = runAgent2.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            action_id = actions.index(action_id)
            done = agent.step(player=2, action_id=action_id)
            if done:
                break

            s, actions = agent.get_actions_space(player=3)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if random.random() < runAgent3.ETA:
                action_id, label = runAgent3.Q.getAction(actions_ont_hot, s)
                if label:
                    runAgent3.SLMemory.append([s, action_id])
            else:
                action_id = runAgent3.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            action_id = actions.index(action_id)
            done = agent.step(player=3, action_id=action_id)
            if done:
                break

                # 每轮更新方法[-1],返回为LR记录类对象列表
            # d1, d2, d3 = agent.get_training_data()
            # 回合更新方法，返回为LR记录类对象列表
        d1, d2, d3 = agent.get_training_data()
        for j in range(len(d1)):
            raw2 = d1[j].a
            hot2 = np.zeros(runAgent1.ACTION_NUM)
            hot2[raw2] = 1
            runAgent1.RLMemory.append([d1[j].s, hot2, d1[j].r, d1[j].s_])
        for j in range(len(d2)):
            raw2 = d2[j].a
            hot2 = np.zeros(runAgent2.ACTION_NUM)
            hot2[raw2] = 1
            runAgent1.RLMemory.append([d2[j].s, hot2, d2[j].r, d2[j].s_])
        for j in range(len(d3)):
            raw2 = d3[j].a
            hot2 = np.zeros(runAgent3.ACTION_NUM)
            hot2[raw2] = 1
            runAgent1.RLMemory.append([d3[j].s, hot2, d3[j].r, d3[j].s_])

        if len(runAgent1.SLMemory) > 500:
            runAgent1.Pi.trainPiNetwork()
        if len(runAgent2.SLMemory) > 500:
            runAgent2.Pi.trainPiNetwork()
        if len(runAgent3.SLMemory) > 500:
            runAgent3.Pi.trainPiNetwork()

        if len(runAgent1.RLMemory) > 500:
            runAgent1.Q.trainQNetwork()
        if len(runAgent2.RLMemory) > 500:
            runAgent2.Q.trainQNetwork()
        if len(runAgent3.RLMemory) > 500:
            runAgent3.Q.trainQNetwork()



