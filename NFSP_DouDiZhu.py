# -*- coding:utf-8 -*-
import agent as ag
import numpy as np
import random
import tensorflow as tf
import RunAgent as RA


if __name__ == '__main__':
    agent = ag.Agent(models=["rl", "rl", "rl"])
    with tf.name_scope('player1'):
        runAgent1 = RA.RunAgent(agent, 'player1')
    with tf.name_scope('player2'):
        runAgent2 = RA.RunAgent(agent, 'player2')
    with tf.name_scope('player3'):
        runAgent3 = RA.RunAgent(agent, 'player3')
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
                action_id, label = runAgent1.Q.getAction(actions_ont_hot, s)
                if label and action_id != 430:
                    SL_in = np.zeros(runAgent1.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent1.SLMemory.append([s, SL_in])
                    runAgent1.Pi.SLMemory = runAgent1.SLMemory
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
            if runAgent2.Q_enable:
                action_id, label = runAgent2.Q.getAction(actions_ont_hot, s)
                if label and action_id != 430:
                    SL_in = np.zeros(runAgent2.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent2.SLMemory.append([s, SL_in])
                    runAgent2.Pi.SLMemory = runAgent2.SLMemory
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
            if runAgent3.Q_enable:
                action_id, label = runAgent3.Q.getAction(actions_ont_hot, s)
                if label and action_id != 430:
                    SL_in = np.zeros(runAgent3.ACTION_NUM)
                    SL_in[action_id] = 1
                    runAgent3.SLMemory.append([s, SL_in])
                    runAgent3.Pi.SLMemory = runAgent3.SLMemory
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
            if count >= 1:
                d1, d2, d3 = agent.get_training_data()
                j = -1
                raw2 = d1[j].a
                hot2 = np.zeros(runAgent1.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d1[j].a_
                hot3 = np.zeros(runAgent1.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430:
                    runAgent1.RLMemory.append([d1[j].s, hot2, d1[j].r, d1[j].s_, hot3])
                raw2 = d2[j].a
                hot2 = np.zeros(runAgent2.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d2[j].a_
                hot3 = np.zeros(runAgent2.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430:
                    runAgent2.RLMemory.append([d2[j].s, hot2, d2[j].r, d2[j].s_, hot3])
                raw2 = d3[j].a
                hot2 = np.zeros(runAgent3.ACTION_NUM)
                hot2[raw2] = 1
                raw3 = d3[j].a_
                hot3 = np.zeros(runAgent3.ACTION_NUM)
                hot3[raw3] = 1
                if raw2 != 430:
                    runAgent3.RLMemory.append([d3[j].s, hot2, d3[j].r, d3[j].s_, hot3])
                runAgent1.Q.REPLAY_MEMORY = runAgent1.RLMemory
                runAgent2.Q.REPLAY_MEMORY = runAgent2.RLMemory
                runAgent3.Q.REPLAY_MEMORY = runAgent3.RLMemory
                if len(runAgent1.RLMemory) == runAgent1.RLMemory_num and len(runAgent1.SLMemory) == runAgent1.SLMemory_num and len(runAgent2.RLMemory) == runAgent2.RLMemory_num and len(runAgent2.SLMemory) == runAgent2.SLMemory_num and len(runAgent3.RLMemory) == runAgent3.RLMemory_num and len(runAgent3.SLMemory) == runAgent3.SLMemory_num:
                    runAgent1.ETA = 0.1
                    runAgent2.ETA = 0.1
                    runAgent3.ETA = 0.1
                    for step in range(runAgent1.Q.Q_step_num):
                        runAgent1.Q.trainQNetwork()
                    print('Episode:', i, ' RL loss of player1:', runAgent1.Q.loss)
                    runAgent1.Q.timeStep = 0
                    for step in range(runAgent1.Pi.timeStep_num):
                        runAgent1.Pi.trainPiNetwork()
                    print('Episode:', i, ' SL loss of player1:', runAgent1.Pi.loss)
                    runAgent1.Pi.timeStep = 0
                    for step in range(runAgent2.Q.Q_step_num):
                        runAgent2.Q.trainQNetwork()
                    print('Episode:', i, ' RL loss of player2:', runAgent2.Q.loss)
                    runAgent2.Q.timeStep = 0
                    for step in range(runAgent2.Pi.timeStep_num):
                        runAgent2.Pi.trainPiNetwork()
                    print('Episode:', i, ' SL loss of player2:', runAgent2.Pi.loss)
                    runAgent2.Pi.timeStep = 0
                    for step in range(runAgent3.Q.Q_step_num):
                        runAgent3.Q.trainQNetwork()
                    print('Episode:', i, ' RL loss of player3:', runAgent3.Q.loss)
                    runAgent3.Q.timeStep = 0
                    for step in range(runAgent3.Pi.timeStep_num):
                        runAgent3.Pi.trainPiNetwork()
                    print('Episode:', i, ' SL loss of player3:', runAgent3.Pi.loss)
                    runAgent3.Pi.timeStep = 0
                else:
                    runAgent1.ETA = 0.5
                    runAgent2.ETA = 0.5
                    runAgent3.ETA = 0.5
                    print('player1 RL memory num:', len(runAgent1.RLMemory), ' SL memory num:', len(runAgent1.SLMemory))
                    print('player2 RL memory num:', len(runAgent2.RLMemory), ' SL memory num:', len(runAgent2.SLMemory))
                    print('player3 RL memory num:', len(runAgent3.RLMemory), ' SL memory num:', len(runAgent3.SLMemory))
            count += 1


        if i % 50 == 1:
            out_file = runAgent1.Agent.game.get_record().records
            out = open('record' + str(i) + '.txt', 'w')
            # print(runAgent1.Agent.game.playrecords.show('=========='), file=out)
            print(out_file, file=out)
            out.close()
            agent_test = ag.Agent(models=["rl", "random", "random"])
            runAgent1.Agent = agent_test
            runAgent1.EPSILON = 0.0
            count_test = 0
            for kk in range(100):
                agent_test.reset()
                done = False
                while not done:
                    s, actions = agent_test.get_actions_space(player=1)
                    actions_ont_hot = np.zeros(agent.dim_actions)
                    for k in range(len(actions)):
                        actions_ont_hot[actions[k]] = 1
                    action_id = runAgent1.Pi.getAction(actions_ont_hot, s)
                    # choose action_id
                    try:
                        action_id = actions.index(action_id)
                    except ValueError:
                        pass
                    done = agent_test.step(player=1, action_id=action_id)
                    winner = agent_test.game.playrecords.winner
                    if winner == 1:
                        count_test += 1
                    if done:
                        break
            print('****************************************** win_rate:', count_test, '% ********************')
            runAgent1.Agent = agent
            runAgent1.EPSILON = 0.01





