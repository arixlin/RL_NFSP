# -*- coding:utf-8 -*-
import DQN_DouDiZhu as DQN
from collections import deque
import AveragePolicyNetwork as SLN
import agent as ag
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
        self.EPISODE_NUM = 5000000


if __name__ == '__main__':
    agent = ag.Agent(models=["rl", "rl", "rl"])
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
                if label and action_id != 429 and action_id != 430:
                    SL_in = np.zeros(runAgent1.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent1.SLMemory.append([s, SL_in])
            else:
                action_id = runAgent1.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            try:
                action_id = actions.index(action_id)
            except ValueError:
                pass
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
                    SL_in = np.zeros(runAgent2.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent2.SLMemory.append([s, SL_in])
            else:
                action_id = runAgent2.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            if action_id == 0:
                pass
            try:
                action_id = actions.index(action_id)
            except ValueError:
                pass
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
                    SL_in = np.zeros(runAgent3.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent3.SLMemory.append([s, SL_in])
            else:
                action_id = runAgent3.Pi.getAction(actions_ont_hot, s)
            # choose action_id
            try:
                action_id = actions.index(action_id)
            except ValueError:
                print(ValueError)
                pass
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
            raw3 = d1[j].a_
            hot3 = np.zeros(runAgent1.ACTION_NUM)
            hot3[raw3] = 1
            if raw2 != 430:
                runAgent1.RLMemory.append([d1[j].s, hot2, d1[j].r, d1[j].s_, hot3])
        for j in range(len(d2)):
            raw2 = d2[j].a
            hot2 = np.zeros(runAgent2.ACTION_NUM)
            hot2[raw2] = 1
            raw3 = d2[j].a_
            hot3 = np.zeros(runAgent2.ACTION_NUM)
            hot3[raw3] = 1
            if raw2 != 430:
                runAgent2.RLMemory.append([d2[j].s, hot2, d2[j].r, d2[j].s_, hot3])
        for j in range(len(d3)):
            raw2 = d3[j].a
            hot2 = np.zeros(runAgent3.ACTION_NUM)
            hot2[raw2] = 1
            raw3 = d3[j].a_
            hot3 = np.zeros(runAgent3.ACTION_NUM)
            hot3[raw3] = 1
            if raw2 != 430:
                runAgent3.RLMemory.append([d3[j].s, hot2, d3[j].r, d3[j].s_, hot3])

        if len(runAgent1.SLMemory) == runAgent1.SLMemory_num:
            runAgent1.Pi.trainPiNetwork('player1')
        if len(runAgent2.SLMemory) == runAgent2.SLMemory_num:
            runAgent2.Pi.trainPiNetwork('player2')
        if len(runAgent3.SLMemory) == runAgent3.SLMemory_num:
            runAgent3.Pi.trainPiNetwork('player3')

        if len(runAgent1.RLMemory) == runAgent1.RLMemory_num:
            runAgent1.Q.trainQNetwork('player1')
        if len(runAgent2.RLMemory) == runAgent2.RLMemory_num:
            runAgent2.Q.trainQNetwork('player2')
        if len(runAgent3.RLMemory) == runAgent3.RLMemory_num:
            runAgent3.Q.trainQNetwork('player3')
        if i % 200 == 1:
            print('=========== episode:', i, '============')
            out_file = runAgent1.Agent.game.get_record().records
            out = open('record' + str(i) + '.txt', 'w')
            print(runAgent1.Agent.game.playrecords.show('=========='))
            print(out_file, file=out)
            out.close()
            agent_test = ag.Agent(models=["rl", "random", "random"])
            runAgent1.Agent = agent_test
            runAgent1.EPSILON = 0.0
            count = 0
            for kk in range(100):
                agent_test.reset()
                done = False
                while not done:
                    s, actions = agent_test.get_actions_space(player=1)
                    actions_ont_hot = np.zeros(agent.dim_actions)
                    for k in range(len(actions)):
                        actions_ont_hot[actions[k]] = 1
                    if random.random() < runAgent1.ETA:
                        action_id, label = runAgent1.Q.getAction(actions_ont_hot, s)
                    else:
                        action_id = runAgent1.Pi.getAction(actions_ont_hot, s)
                    # choose action_id
                    try:
                        action_id = actions.index(action_id)
                    except ValueError:
                        pass
                    done = agent_test.step(player=1, action_id=action_id)
                    winner = agent_test.game.playrecords.winner
                    if winner == 1:
                        count += 1
                    if done:
                        break
            print('****************************************** win_rate:', count, '% ********************')
            runAgent1.Agent = agent
            runAgent1.EPSILON = 0.01






