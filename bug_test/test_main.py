import numpy as np
import random
import AveragePolicyNetwork as P
import DQN_DouDiZhu as Q

if __name__ == '__main__':
    SLMemory = []
    for i in range(200):
        state = np.ones(33) * random.random()
        action = np.ones(431) * random.random()
        SLMemory.append([state, action])

    net1 = P.Pi(431, 33, SLMemory, 200, 'player1')
    net2 = P.Pi(431, 33, SLMemory, 200, 'player2')
    net3 = P.Pi(431, 33, SLMemory, 200, 'player3')

    for i in range(50):
        net1.trainPiNetwork()
        print('Episode:', i, ' SL loss of player1:', net1.loss)

        net2.trainPiNetwork()
        print('Episode:', i, ' SL loss of player2:', net2.loss)

        net3.trainPiNetwork()
        print('Episode:', i, ' SL loss of player3:', net3.loss)
