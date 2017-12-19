import numpy as np
from agent import Agent

def prn_obj(obj):
    return (', '.join([''"'%s'"':'"'%s'"'' % item for item in obj.__dict__.items()]))


def object2list(obj):
    memoryRL = []
    for sample in obj:
        memoryRL_sample = [[] for i in range(5)]
        for name, value in sample.__dict__.items():
            #a(t+1) 游戏结束, 没有后续行为
            if value is None:
                value = [-1]
            if name == 's':
                memoryRL_sample[0].append(value)
            elif name == 'a':
                memoryRL_sample[1].append(value)
            elif name == 'r':
                memoryRL_sample[2].append(value)
            elif name == 's_':
                memoryRL_sample[3].append(value)
            elif name == 'a_':
                memoryRL_sample[4].append(value)
        memoryRL.append(memoryRL_sample)
    return memoryRL


# rl
def doudizhu_agent():
    Mrl = []
    agent = Agent(models=["rl", "rl", "rl"])
    agent.reset()
    done = False
    while (True):
        s, actions = agent.get_actions_space(player=1)
        # print(len(actions))

        # choose action_id
        done = agent.step(player=1, action_id=0)
        # print(done)
        if done:
            break

        s, actions = agent.get_actions_space(player=2)
        # choose action_id
        done = agent.step(player=2, action_id=0)
        if done:
            break

        s, actions = agent.get_actions_space(player=3)
        # choose action_id
        done = agent.step(player=3, action_id=0)
        if done:
            break

        # 每轮更新方法[-1],返回为LR记录类对象列表
        # d1, d2, d3 = agent.get_training_data()

    # 回合更新方法，返回为LR记录类对象列表
    d1, d2, d3 = agent.get_training_data()
    d1_memoryRL = object2list(d1)
    d2_memoryRL = object2list(d2)
    d3_memoryRL = object2list(d3)
    Mrl.append(d1_memoryRL)
    Mrl.append(d2_memoryRL)
    Mrl.append(d3_memoryRL)
    # a = prn_obj(d1[1])
    # print(a)
    # print(eval('{' + a.replace('\n', ',') + '}')['s_'])
    # winner = agent.game.playrecords.winner

    # print(agent.dim_states)
    # print(agent.dim_actions)
    return Mrl

doudizhu_agent()
