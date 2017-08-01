import numpy as np
import tensorflow as tf
import random

class Pi:
    """class for average-policy network"""
    def __init__(self, ACTION_NUM, STATE_NUM, SLMemory, SLMemory_num, player):
        self.train_phase = False
        self.player = player
        self.ACTION_NUM = ACTION_NUM
        self.STATE_NUM = STATE_NUM
        self.SLMemory = SLMemory
        self.BATCH_SIZE = 16
        self.timeStep = 0
        self.timeStep_num = 10
        self.createPiNetwork()
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

    def createPiNetwork(self):
        # input layer
        def conv_layer(x, W, b, stride):
            conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, b)
            norm = tf.nn.lrn(conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            return norm

        self.stateInput = tf.placeholder(tf.float32, shape=[None, 163, 144, 1])
        self.actionOutput = tf.placeholder(tf.float32, shape=[None, self.ACTION_NUM])
        self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')

        # weights
        weights = {
            'W1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.02), trainable=True, name='W1'),
            'W2': tf.Variable(tf.truncated_normal([5, 5, 64, 192], stddev=0.02), trainable=True, name='W2'),
            'W3': tf.Variable(tf.truncated_normal([3, 3, 192, 256], stddev=0.02), trainable=True, name='W3'),
            'W_fc6': tf.Variable(tf.truncated_normal([19 * 17 * 256, 2048], stddev=0.02), trainable=True, name='W_fc6'),
            'W_fc8': tf.Variable(tf.truncated_normal([2048, self.ACTION_NUM], stddev=0.02), trainable=True, name='W_fc8')
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name='b1'),
            'b2': tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name='b2'),
            'b3': tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name='b3'),
            'b_fc6': tf.Variable(tf.constant(0.0, shape=[2048]), trainable=True, name='b_fc6'),
            'b_fc8': tf.Variable(tf.constant(0.0, shape=[self.ACTION_NUM]), trainable=True, name='b_fc8')
        }

        with tf.name_scope('conv1'):
            conv1 = conv_layer(self.stateInput, weights['W1'], biases['b1'], 1)
            conv1 = tf.nn.relu(conv1)

        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        with tf.name_scope('conv2'):
            conv2 = conv_layer(pool1, weights['W2'], biases['b2'], 1)
            conv2 = tf.nn.relu(conv2)

        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        with tf.name_scope('conv3'):
            conv3 = conv_layer(pool2, weights['W3'], biases['b3'], 1)
            conv3 = tf.nn.relu(conv3)

        with tf.name_scope('pool5'):
            pool5 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        with tf.name_scope('fc6'):
            pool5_flat = tf.reshape(pool5, shape=[-1, 19 * 17 * 256])
            fc6 = tf.nn.relu(tf.matmul(pool5_flat, weights['W_fc6']) + biases['b_fc6'], name='fc6')

        with tf.name_scope('drop6'):
            drop6 = tf.nn.dropout(fc6, keep_prob=self.keep_probability)

        with tf.name_scope('fc8'):
            self.output = tf.add(tf.matmul(drop6, weights['W_fc8']), biases['b_fc8'], name='fc8')

        # layers
        # h_layer1 = self.batch_norm(h_layer1)
        self.out = tf.nn.softmax(self.output)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.actionOutput, logits=self.output))
        self.trainStep = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + self.player + '/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            self.session.run(tf.initialize_all_variables())


    def trainPiNetwork(self):
        # Pi_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.player)
        # for var in Pi_var_list:
        #     if 'Pi' in var.name:
        #         print('pre ' + var.name)
        #         print(self.session.run(var.name))
        minibatch = random.sample(self.SLMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        # state_batch = np.zeros([5, 33])
        action_batch = [data[1] for data in minibatch]
        # action_batch = np.zeros([5, 431])
        # if self.player == 'player3':
        #     state_batch = np.ones([5, 33])
        #     action_batch = np.ones([5, 431])

        # print('=============')
        # print(action_batch)

        # if self.timeStep == 0:
        #     checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + self.player + '/')
        #     if checkpoint and checkpoint.model_checkpoint_path:
        #         self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        #       # print('model loaded')

        self.session.run(self.trainStep, feed_dict={
            self.actionOutput: action_batch,
            self.stateInput: state_batch,
            self.keep_probability: 0.7
        })
        self.loss = self.session.run(self.cost, feed_dict={
            self.actionOutput: action_batch,
            self.stateInput: state_batch,
            self.keep_probability: 1.0
        })

        if self.total_step % 2000 == 1:
        # if self.timeStep == self.timeStep_num - 1:
            self.saver.save(self.session, 'saved_PiNetworks_' + self.player + '/model.ckpt')
        # print('model saved')
        self.timeStep += 1
        self.total_step += 1
        # for var in Pi_var_list:
        #     if 'Pi' in var.name:
        #     	print('after ' + var.name)
        #     	print(self.session.run(var.name))

    def getAction(self, action_space, state):
        # checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + self.player + '/')
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        #     # print('model loaded')
        self.train_phase = False
        # state = np.zeros(33)
        state = np.expand_dims(state, -1)
        self.QValue = self.session.run(self.out, feed_dict={self.stateInput: [state], self.keep_probability: 1.0})[0]
        Q_test = self.QValue * action_space
        # print('Qtest ' + self.player)
        # print(Q_test)
        if max(Q_test) <= 0.0000001:
            action_index = random.randrange(self.ACTION_NUM)
            while action_space[action_index] != 1:
                action_index = random.randrange(self.ACTION_NUM)
        else:
            action_index = np.argmax(self.QValue * action_space)
        # if self.QValue[action_index] <= 0.0:
        #     action_index = random.randrange(self.ACTION_NUM)
        #     while action_space[action_index] != 1:
        #         action_index = random.randrange(self.ACTION_NUM)
        return action_index
