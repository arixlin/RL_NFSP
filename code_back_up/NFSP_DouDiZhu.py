# -*- coding:utf-8 -*-
import DQN_DouDiZhu as DQN
from collections import deque
import AveragePolicyNetwork as SLN
import agent as ag
import numpy as np
import random


class RunAgent:
    """class for an agent"""
    def __init__(self, agent, player):
        self.Agent = agent
        self.player = player
        self.ACTION_NUM = agent.dim_actions
        self.STATE_NUM = agent.dim_states
        self.RLMemory_num = 20
        self.SLMemory_num = 20
        self.RLMemory = deque(maxlen=self.RLMemory_num)
        self.SLMemory = deque(maxlen=self.SLMemory_num)
        # self.Q = DQN.DQN_DouDiZhu(self.ACTION_NUM, self.STATE_NUM, self.RLMemory, self.RLMemory_num, self.player)
        # self.Pi = SLN.Pi(self.ACTION_NUM, self.STATE_NUM, self.SLMemory, self.SLMemory_num, self.player)
        self.EPSILON = 0.06
        self.ETA = 0.1
        self.EPISODE_NUM = 5000000
        self.Q_enable = False


if __name__ == '__main__':
    agent = ag.Agent(models=["rl", "rl", "rl"])
    runAgent1 = RunAgent(agent, 'player1')
    runAgent2 = RunAgent(agent, 'player2')
    runAgent3 = RunAgent(agent, 'player3')
    Q = DQN.DQN_DouDiZhu(runAgent1.ACTION_NUM, runAgent1.STATE_NUM, runAgent1.RLMemory, runAgent1.RLMemory_num)
    Pi = SLN.Pi(runAgent1.ACTION_NUM, runAgent1.STATE_NUM, runAgent1.SLMemory, runAgent1.SLMemory_num)

    for i in range(runAgent1.EPISODE_NUM):
        print('=========== episode:', i, '============')
        if random.random() < runAgent1.ETA:
            runAgent1.Q_enable = True
            print('player1 ' + 'Q network is working')
        else:
            runAgent1.Q_enable = False
            print('player1 ' + 'Pi network is working')

        if random.random() < runAgent2.ETA:
            runAgent2.Q_enable = True
            print('player2 ' + 'Q network is working')
        else:
            runAgent2.Q_enable = False
            print('player2 ' + 'Pi network is working')

        if random.random() < runAgent3.ETA:
            runAgent3.Q_enable = True
            print('player3 ' + 'Q network is working')
        else:
            runAgent3.Q_enable = False
            print('player3 ' + 'Pi network is working')

        agent.reset()
        done = False
        count = 0
        while(True):
            s, actions = agent.get_actions_space(player=1)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if runAgent1.Q_enable:
                Q.createQNetwork('player1')
                action_id, label = Q.getAction(actions_ont_hot, s, 'player1')
                if label and action_id != 430 and action_id != 429:
                    SL_in = np.zeros(runAgent1.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent1.SLMemory.append([s, SL_in])
            else:
                Pi.createPiNetwork('player1')
                action_id = Pi.getAction(actions_ont_hot, s, 'player1')
            # choose action_id
            try:
                action_id = actions.index(action_id)
            except ValueError:
                print(ValueError)
                exit()
            done = agent.step(player=1, action_id=action_id)
            if done:
                break

            s, actions = agent.get_actions_space(player=2)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if runAgent2.Q_enable:
                Q.createQNetwork('player2')
                action_id, label = Q.getAction(actions_ont_hot, s, 'player2')
                if label and action_id != 430 and action_id != 429:
                    SL_in = np.zeros(runAgent2.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent2.SLMemory.append([s, SL_in])
            else:
                Pi.createPiNetwork('player2')
                action_id = Pi.getAction(actions_ont_hot, s, 'player2')
            # choose action_id
            if action_id == 0:
                pass
            try:
                action_id = actions.index(action_id)
            except ValueError:
                print(ValueError)
                exit()
            done = agent.step(player=2, action_id=action_id)
            if done:
                break

            s, actions = agent.get_actions_space(player=3)
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
            if runAgent3.Q_enable:
                Q.createQNetwork('player3')
                action_id, label = Q.getAction(actions_ont_hot, s)
                if label and action_id != 430 and action_id != 429:
                    SL_in = np.zeros(runAgent3.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent3.SLMemory.append([s, SL_in])
            else:
                Pi.createPiNetwork('player3')
                action_id = Pi.getAction(actions_ont_hot, s, 'player3')
            # choose action_id
            try:
                action_id = actions.index(action_id)
            except ValueError:
                print(ValueError)
                exit()
            done = agent.step(player=3, action_id=action_id)
            if done:
                break

                # 每轮更新方法[-1],返回为LR记录类对象列表
            if count >= 1:
                d1, d2, d3 = agent.get_training_data()
                j = -1
                raw2 = d1[j].a
                hot2 = np.zeros(runAgent1.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d1[j].a_
                hot3 = np.zeros(runAgent1.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430 and raw2 != 429:
                    runAgent1.RLMemory.append([d1[j].s, hot2, d1[j].r, d1[j].s_, hot3])
                raw2 = d2[j].a
                hot2 = np.zeros(runAgent2.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d2[j].a_
                hot3 = np.zeros(runAgent2.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430 and raw2 != 429:
                    runAgent2.RLMemory.append([d2[j].s, hot2, d2[j].r, d2[j].s_, hot3])
                raw2 = d3[j].a
                hot2 = np.zeros(runAgent3.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d3[j].a_
                hot3 = np.zeros(runAgent3.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430 and raw2 != 429:
                    runAgent3.RLMemory.append([d3[j].s, hot2, d3[j].r, d3[j].s_, hot3])
                if len(runAgent1.RLMemory) == runAgent1.RLMemory_num and len(runAgent1.SLMemory) == runAgent1.SLMemory_num and len(runAgent2.RLMemory) == runAgent2.RLMemory_num and len(runAgent2.SLMemory) == runAgent2.SLMemory_num and len(runAgent3.RLMemory) == runAgent3.RLMemory_num and len(runAgent3.SLMemory) == runAgent3.SLMemory_num:
                    runAgent1.ETA = 0.1
                    runAgent2.ETA = 0.1
                    runAgent3.ETA = 0.1
                    # runAgent1.ETA = 0
                    # runAgent2.ETA = 0
                    # runAgent3.ETA = 0
                    # for step in range(runAgent1.Q.Q_step_num):
                        # runAgent1.Q.trainQNetwork()
                    # print('Episode:', i, ' RL loss of player1:', runAgent1.Q.loss)
                    # runAgent1.Q.timeStep = 0
                    for step in range(runAgent1.Pi.timeStep_num):
                        Pi.createPiNetwork('player1')
                        Pi.trainPiNetwork('player1')
                    print('Episode:', i, ' SL loss of player1:', Pi.loss)
                    runAgent1.Pi.timeStep = 0
                    # for step in range(runAgent2.Q.Q_step_num):
                        # runAgent2.Q.trainQNetwork()
                    # print('Episode:', i, ' RL loss of player2:', runAgent2.Q.loss)
                    # runAgent2.Q.timeStep = 0
                    for step in range(runAgent2.Pi.timeStep_num):
                        Pi.createPiNetwork('player2')
                        Pi.trainPiNetwork('player2')
                    print('Episode:', i, ' SL loss of player2:', Pi.loss)
                    runAgent2.Pi.timeStep = 0
                    # for step in range(runAgent3.Q.Q_step_num):
                        # runAgent3.Q.trainQNetwork()
                    # print('Episode:', i, ' RL loss of player3:', runAgent3.Q.loss)
                    # runAgent3.Q.timeStep = 0
                    for step in range(runAgent3.Pi.timeStep_num):
                        Pi.createPiNetwork('player3')
                        Pi.trainPiNetwork('player3')
                    print('Episode:', i, ' SL loss of player3:', Pi.loss)
                    runAgent3.Pi.timeStep = 0
                else:
                    runAgent1.ETA = 0.5
                    runAgent2.ETA = 0.5
                    runAgent3.ETA = 0.5
                    print('player1 RL memory num:', len(runAgent1.RLMemory), ' SL memory num:', len(runAgent1.SLMemory))
                    print('player2 RL memory num:', len(runAgent2.RLMemory), ' SL memory num:', len(runAgent2.SLMemory))
                    print('player3 RL memory num:', len(runAgent3.RLMemory), ' SL memory num:', len(runAgent3.SLMemory))
            count += 1

            # 回合更新方法，返回为LR记录类对象列表

        # if i % 50 == 1:
        #     out_file = runAgent1.Agent.game.get_record().records
        #     out = open('record' + str(i) + '.txt', 'w')
        #     # print(runAgent1.Agent.game.playrecords.show('=========='), file=out)
        #     print(out_file, file=out)
        #     out.close()
        #     agent_test = ag.Agent(models=["rl", "random", "random"])
        #     runAgent1.Agent = agent_test
        #     runAgent1.EPSILON = 0.0
        #     count_test = 0
        #     for kk in range(100):
        #         agent_test.reset()
        #         done = False
        #         while not done:
        #             s, actions = agent_test.get_actions_space(player=1)
        #             actions_ont_hot = np.zeros(agent.dim_actions)
        #             for k in range(len(actions)):
        #                 actions_ont_hot[actions[k]] = 1
        #             action_id = runAgent1.Pi.getAction(actions_ont_hot, s)
        #             # choose action_id
        #             try:
        #                 action_id = actions.index(action_id)
        #             except ValueError:
        #                 pass
        #             done = agent_test.step(player=1, action_id=action_id)
        #             winner = agent_test.game.playrecords.winner
        #             if winner == 1:
        #                 count_test += 1
        #             if done:
        #                 break
        #     print('****************************************** win_rate:', count_test, '% ********************')
        #     runAgent1.Agent = agent
        #     runAgent1.EPSILON = 0.01






