import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN_DouDiZhu:
    """DQN part of NFSP"""
    def __init__(self, ACTION_NUM, STATE_NUM, REPLAY_MEMORY):
        self.ACTION_NUM = ACTION_NUM
        self.STATE_NUM = STATE_NUM
        self.EPSILON = 0.1
        self.GAMMA = 0.9
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.BATCH_SIZE = 32
        self.createQNetwork()
        self.timeStep = 0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def createQNetwork(self):
        # input layer
        self.stateInput = tf.placeholder(dtype=tf.float32, shape=[None, self.STATE_NUM])
        self.actionInput = tf.placeholder(dtype=tf.float32, shape=[None, self.ACTION_NUM])
        self.yInput = tf.placeholder(dtype=tf.float32, shape=[None])

        # weights
        W1 = self.weight_variable([self.STATE_NUM, 256])
        b1 = self.bias_variable([256])

        W2 = self.weight_variable([256, 512])
        b2 = self.bias_variable([512])

        W3 = self.weight_variable([512, self.ACTION_NUM])
        b3 = self.bias_variable([self.ACTION_NUM])

        # layers
        h_layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.stateInput, W1), b1))
        h_layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_layer1, W2), b2))
        self.QValue = tf.nn.bias_add(tf.matmul(h_layer2, W3), b3)
        Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_DQNetworks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def trainQNetwork(self, player):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.REPLAY_MEMORY, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValue.eval(feed_dict={self.stateInput: nextState_batch})
        for i in range(0, self.BATCH_SIZE):
            terminal = minibatch[i][2]
            if terminal != 0:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })
        print(player + '_' + 'RL_step:', self.timeStep, ' ', 'RL_loss:', self.cost.eval(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        }))

        # save network every 100000 iteration
        if self.timeStep % 100 == 0:
            self.saver.save(self.session, 'saved_Networks/' + 'network' + '-dqn', global_step=self.timeStep)

        self.timeStep += 1

    def getAction(self, action_space, state):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [state]})[0]
        label = False
        if random.random() <= self.EPSILON:
            action_index = random.randrange(self.ACTION_NUM)
            while action_space[action_index] != 1:
                action_index = random.randrange(self.ACTION_NUM)
        else:
            Q_test = QValue * action_space
            if max(Q_test) <= 0.0000001:
                action_index = random.randrange(self.ACTION_NUM)
                while action_space[action_index] != 1:
                    action_index = random.randrange(self.ACTION_NUM)
                label = False
            else:
                action_index = np.argmax(QValue * action_space)
                label = True
            # if QValue[action_index] <= 0.0:
            #     action_index = random.randrange(self.ACTION_NUM)
            #     while action_space[action_index] != 1:
            #         action_index = random.randrange(self.ACTION_NUM)
            #     label = False
        return action_index, label
