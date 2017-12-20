import numpy as np
from agent import Agent

def prn_obj(obj):
    return (', '.join([''"'%s'"':'"'%s'"'' % item for item in obj.__dict__.items()]))


def object2list(obj):
    memoryRL = []
    for sample in obj:
        memoryRL_sample = [[] for i in range(5)]
        memoryRL_sample[0] = sample.s
        memoryRL_sample[1] = sample.a
        memoryRL_sample[2] = sample.r
        memoryRL_sample[3] = sample.s_
        # a(t+1) 游戏结束, 没有后续行为
        if sample.a_ is None:
            memoryRL_sample[4] = -1
        else:
            memoryRL_sample[4] = sample.a_
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

    print(Mrl)
    # print(agent.dim_states)
    # print(agent.dim_actions)
    return Mrl

doudizhu_agent()
