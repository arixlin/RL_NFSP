import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN_DouDiZhu:
    """DQN part of NFSP"""
    def __init__(self, ACTION_NUM, STATE_NUM, REPLAY_MEMORY, REPLAY_MEMORY_NUM, player):
        self.train_phase = False
        self.player = player
        self.ACTION_NUM = ACTION_NUM
        self.STATE_NUM = STATE_NUM
        self.EPSILON = 0.1
        self.GAMMA = 0.95
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.BATCH_SIZE = 32
        self.timeStep = 0
        self.Q_step_num = 3
        self.createQNetwork()
        self.total_step = 0

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.get_variable(name=name, initializer=initial, trainable=True)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=True)

    def batch_norm(self, X):
        train_phase = self.train_phase
        with tf.name_scope('bn'):
            n_out = X.get_shape()[-1:]
            beta = tf.Variable(tf.constant(0.0, shape=n_out), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=n_out), name='gamma', trainable=True)
            # batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments')
            batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(train_phase, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
        return normed


    def createQNetwork(self):
        # input layer
        def conv_layer(x, W, b, stride):
            conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, b)
            norm = tf.nn.lrn(conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            return norm

        self.stateInput = tf.placeholder(dtype=tf.float32, shape=[None, 163, 144, 1])
        self.actionInput = tf.placeholder(dtype=tf.float32, shape=[None, self.ACTION_NUM])
        self.yInput = tf.placeholder(dtype=tf.float32, shape=[None])
        self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')

        # weights
        weights = {
            'W1': tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.02), trainable=True, name='W1'),
            'W2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.02), trainable=True, name='W2'),
            'W3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.02), trainable=True, name='W3'),
            'W_fc6': tf.Variable(tf.truncated_normal([4 * 4 * 128, 1024], stddev=0.02), trainable=True, name='W_fc6'),
            'W_fc8': tf.Variable(tf.truncated_normal([1024, self.ACTION_NUM], stddev=0.02), trainable=True, name='W_fc8')
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.0, shape=[32]), trainable=True, name='b1'),
            'b2': tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name='b2'),
            'b3': tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name='b3'),
            'b_fc6': tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name='b_fc6'),
            'b_fc8': tf.Variable(tf.constant(0.0, shape=[self.ACTION_NUM]), trainable=True, name='b_fc8')
        }

        with tf.name_scope('conv1'):
            conv1 = conv_layer(self.stateInput, weights['W1'], biases['b1'], 1)
            conv1 = tf.nn.relu(conv1)

        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        with tf.name_scope('conv2'):
            conv2 = conv_layer(pool1, weights['W2'], biases['b2'], 2)
            conv2 = tf.nn.relu(conv2)

        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        with tf.name_scope('conv3'):
            conv3 = conv_layer(pool2, weights['W3'], biases['b3'], 2)
            conv3 = tf.nn.relu(conv3)

        with tf.name_scope('pool5'):
            pool5 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        with tf.name_scope('fc6'):
            pool5_flat = tf.reshape(pool5, shape=[-1, 4 * 4 * 128])
            fc6 = tf.nn.relu(tf.matmul(pool5_flat, weights['W_fc6']) + biases['b_fc6'], name='fc6')

        with tf.name_scope('drop6'):
            drop6 = tf.nn.dropout(fc6, keep_prob=self.keep_probability)

        with tf.name_scope('fc8'):
            self.QValue = tf.add(tf.matmul(drop6, weights['W_fc8']), biases['b_fc8'], name='fc8')
        # h_layer1 = self.batch_norm(h_layer1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - tf.reduce_sum(self.QValue * self.actionInput, axis=1)))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        if self.player == 'player1':
            checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_' + self.player + '/')
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
                self.session.run(tf.initialize_all_variables())
        else:
            checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_' + 'player1' + '/')
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path, 'of player1 Q')
            else:
                print("Could not find old network weights of player1 Q")
                self.session.run(tf.initialize_all_variables())

    def trainQNetwork(self):
        self.train_phase = True
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.REPLAY_MEMORY, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        next_action_batch = [data[4] for data in minibatch]

        if self.timeStep == 0:
            # checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_new_' + self.player + '/')
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            # self.saver.save(self.session, 'saved_QNetworks_old_' + self.player + '/model_old.ckpt')
            # print('old model replaced successfully!')

            # checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_' + self.player + '/')
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            # print('old model loaded')
            self.QValue_batch = self.session.run(self.QValue, feed_dict={self.stateInput: nextState_batch, self.keep_probability: 1.0})
            # self.QValue_batch = tf.stop_gradient(self.QValue_batch)
            # checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_new_' + self.player + '/')
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            # print('new model loaded')

        # Step 2: calculate y
        y_batch = []
        for i in range(0, self.BATCH_SIZE):
            terminal = minibatch[i][2]
            if terminal != 0:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.GAMMA * np.max(self.QValue_batch[i]))

        self.session.run(self.trainStep, feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch,
            self.keep_probability: 0.7
        })
        self.loss = self.session.run(self.cost, feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch,
                self.keep_probability: 1.0
            })

        # save network every 100000 iteration
        if self.total_step % 100 == 1:
        # if self.timeStep == self.Q_step_num - 1:
            self.saver.save(self.session, 'saved_QNetworks_' + self.player + '/model.ckpt')
            # print('new model saved')
        self.timeStep += 1
        self.total_step += 1

    def getAction(self, action_space, state):
        # checkpoint = tf.train.get_checkpoint_state('saved_QNetworks_new_' + self.player + '/')
        # if checkpoint and checkpoint.model_checkpoint_path:
            # self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        # print('new model loaded')
        self.train_phase = False
        state = np.expand_dims(state, -1)
        QValue = self.session.run(self.QValue, feed_dict={self.stateInput: [state], self.keep_probability: 1.0})[0]
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
