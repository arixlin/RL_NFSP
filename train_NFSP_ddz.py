# -*- coding:utf-8 -*-    import agent as ag
import agent as ag
import numpy as np
import random
import tensorflow as tf
import RunAgent as RA
import DQN_DouDiZhu as Q
import AveragePolicyNetwork as Pi

#把actions分行存在state矩阵的最后三行
def combine(s, a):
    #get s cloumn
    dim = s.shape[1]
    s[-3, :] = a[:dim]
    s[-2, :] = a[dim: 2 * dim]
    s[-1, :-1] = a[2 * dim: ]
    return s

def test_Pi_network(runAgent1):
    agent_test = ag.Agent(models=["rl", "rl", "rl"])
    runAgent1.Agent = agent_test
    count_test = 0
    for kk in range(100):
        agent_test.reset()
        done = False
        # play 100 times, count player-1 win times
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
    # out_file = runAgent1.Agent.game.get_record().records
    # out = open('records/random_test_record' + str(i) + '.txt', 'w')
    # print(out_file, file=out)
    # out.close()
    print('******************* win_rate_with_random:', count_test, '% ********************')


def action_one_hot(actions, agent):
    actions_ont_hot = np.zeros(agent.dim_actions)
    for k in range(len(actions)):
        actions_ont_hot[actions[k]] = 1.0
    return actions_ont_hot


if __name__ == '__main__':
    agent = ag.Agent(models=["rl", "rl", "rl"])
    runAgent1 = RA.RunAgent(agent, 'player1')

    q_net = Q.DQN_DouDiZhu(ACTION_NUM=runAgent1.ACTION_NUM, STATE_NUM=runAgent1.STATE_NUM,
                           REPLAY_MEMORY=runAgent1.RLMemory, REPLAY_MEMORY_NUM=runAgent1.RLMemory_num, player='player_past')
    pi_net = Pi.Pi(ACTION_NUM=runAgent1.ACTION_NUM, STATE_NUM=runAgent1.STATE_NUM,
                   SLMemory=runAgent1.SLMemory, SLMemory_num=runAgent1.SLMemory_num, player='player_past')
    train_count = 0
    win_count = 0

    for i in range(runAgent1.EPISODE_NUM):
        train_count += 1
        print('=========== episode:', i, '============')
        if random.random() < runAgent1.ETA:
            runAgent1.Q_enable = True
            print('player1 ' + 'Q network is working')
        else:
            runAgent1.Q_enable = False
            print('player1 ' + 'Pi network is working')

        #player state init start_state
        agent.reset()
        #game over label
        done = False
        count = 0

        while(True):
            #player1
            #get game state and player next_step actions
            s, actions = agent.get_actions_space(player=1)
            #number action lable to one hot label
            actions_ont_hot = action_one_hot(actions, agent)
            #Q learing or AveragePolicy
            if runAgent1.Q_enable:
                #get biggest probability action id
                action_id, label = runAgent1.Q.getAction(actions_ont_hot, s)

                #lable == True  430: can't match, yaobuqi
                if label and action_id != 430:
                    SL_in = np.zeros(runAgent1.ACTION_NUM, dtype=np.float32)
                    SL_in[action_id] = 1.0
                    # 429: pass, buyao
                    if action_id == 429:
                        print('alert!')

                    s = combine(s, SL_in)
                    s = np.expand_dims(s, -1)
                    runAgent1.SLMemory.append([s, SL_in])
                    runAgent1.Pi.SLMemory = runAgent1.SLMemory
            else:
                action_id = runAgent1.Pi.getAction(actions_ont_hot.astype(np.float32), s.astype(np.float32))
            # choose action_id
            action_id = actions.index(action_id)
            #execute id
            done = agent.step(player=1, action_id=action_id)
            if done:
                win_count += 1
                break

            #player2
            s, actions = agent.get_actions_space(player=2)
            actions_ont_hot = action_one_hot(actions, agent)
            action_id = pi_net.getAction(actions_ont_hot.astype(np.float32), s.astype(np.float32))
            # choose action_id
            action_id = actions.index(action_id)
            done = agent.step(player=2, action_id=action_id)
            if done:
                break

            #player3
            s, actions = agent.get_actions_space(player=3)
            actions_ont_hot = action_one_hot(actions, agent)
            action_id = pi_net.getAction(actions_ont_hot, s)
            # choose action_id
            action_id = actions.index(action_id)
            done = agent.step(player=3, action_id=action_id)
            if done:
                break

            # 每轮更新方法[-1],返回为LR记录类对象列表,
            if count >= 1:
                d1, d2, d3 = agent.get_training_data()
                j = -1
                raw2 = d1[j].a
                hot2 = np.zeros(runAgent1.ACTION_NUM, dtype=np.float32)
                hot2[raw2] = 1.0
                s = combine(d1[j].s, hot2)
                s = np.expand_dims(s, -1)
                raw3 = d1[j].a_
                hot3 = np.zeros(runAgent1.ACTION_NUM, dtype=np.float32)
                hot3[raw3] = 1.0
                s_ = combine(d1[j].s_, hot3)
                s_ = np.expand_dims(s_, -1)
                if raw2 != 430:
                    runAgent1.RLMemory.append([s, hot2.astype(np.float32), np.float32(d1[j].r), s_, hot3.astype(np.float32)])

                runAgent1.Q.REPLAY_MEMORY = runAgent1.RLMemory

                t_d_len = False
                #如果获取到200条训练数据则开始训练网络
                if len(runAgent1.RLMemory) == runAgent1.RLMemory_num and len(runAgent1.SLMemory) == runAgent1.SLMemory_num:
                    runAgent1.ETA = 0.1
                    # runAgent2.ETA = 0.1
                    # runAgent3.ETA = 0.1
                    for step in range(runAgent1.Q.Q_step_num):
                        runAgent1.Q.trainQNetwork()
                    # print('Episode:', i, ' RL loss of player1:', runAgent1.Q.loss)
                    runAgent1.Q.timeStep = 0
                    for step in range(runAgent1.Pi.timeStep_num):
                        runAgent1.Pi.trainPiNetwork()
                    # print('Episode:', i, ' SL loss of player1:', runAgent1.Pi.loss)
                    runAgent1.Pi.timeStep = 0
                    t_d_len = True
                else:
                    runAgent1.ETA = 0.5
                    # print('player1 RL memory num:', len(runAgent1.RLMemory), ' SL memory num:', len(runAgent1.SLMemory))
            count += 1

        if t_d_len:
            print('Episode:', i, ' RL loss of player1:', runAgent1.Q.loss)
            print('Episode:', i, ' SL loss of player1:', runAgent1.Pi.loss)

        if i % 20 == 0:
            test_Pi_network(runAgent1)
            runAgent1.Agent = agent

        if train_count == 500:
            print('******************* win_rate_with_past_self:', win_count, '% ********************')
            q_net.weights_saver.restore(q_net.session, 'saved_QNetworks_past/weights.ckpt')
            q_net.biases_saver.restore(q_net.session, 'saved_QNetworks_past/biases.ckpt')
            print('Player_past successfully loaded weights of player1 Q')
            pi_net.weights_saver.restore(pi_net.session, 'saved_PiNetworks_past/weights.ckpt')
            pi_net.biases_saver.restore(pi_net.session, 'saved_PiNetworks_past/biases.ckpt')
            print('Player_past successfully loaded biases of player1 Pi')

            train_count = 0
            win_count = 0
            runAgent1.SLMemory.clear()
            runAgent1.RLMemory.clear()






