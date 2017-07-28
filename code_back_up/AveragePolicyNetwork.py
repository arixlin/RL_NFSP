import numpy as np
import tensorflow as tf
import random

class Pi:
    """class for average-policy network"""
    def __init__(self, ACTION_NUM, STATE_NUM, SLMemory, SLMemory_num):
        self.train_phase = False
        self.ACTION_NUM = ACTION_NUM
        self.STATE_NUM = STATE_NUM
        self.SLMemory = SLMemory
        self.BATCH_SIZE = 5
        self.timeStep = 0
        self.timeStep_num = 5

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

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

    def createPiNetwork(self, player):
        # input layer
        self.stateInput = tf.placeholder(tf.float32, shape=[None, self.STATE_NUM])
        self.actionOutput = tf.placeholder(tf.float32, shape=[None, self.ACTION_NUM])

        # weights
        W1 = self.weight_variable([self.STATE_NUM, 256])
        b1 = self.bias_variable([256])

        W2 = self.weight_variable([256, 512])
        b2 = self.bias_variable([512])

        W3 = self.weight_variable([512, self.ACTION_NUM])
        b3 = self.bias_variable([self.ACTION_NUM])

        # layers
        h_layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.stateInput, W1), b1))
        # h_layer1 = self.batch_norm(h_layer1)
        h_layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_layer1, W2), b2))
        # h_layer2 = self.batch_norm(h_layer2)
        self.output = tf.nn.bias_add(tf.matmul(h_layer2, W3), b3)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.actionOutput, logits=self.output))
        self.out = tf.nn.softmax(self.output)
        self.trainStep = tf.train.GradientDescentOptimizer(1e-2).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + player + '/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            self.session.run(tf.initialize_all_variables())

    def trainPiNetwork(self, player):
        self.train_phase = True
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.SLMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        # print('=============')
        # print(action_batch)

        checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + player + '/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            # print('model loaded')

        self.trainStep.run(feed_dict={
            self.actionOutput: action_batch,
            self.stateInput: state_batch
        })
        self.loss = self.cost.eval(feed_dict={
            self.actionOutput: action_batch,
            self.stateInput: state_batch
        })

        self.saver.save(self.session, 'saved_PiNetworks_' + player + '/model.ckpt')
        # print('model saved')
        self.timeStep += 1

    def getAction(self, action_space, state, player):
        checkpoint = tf.train.get_checkpoint_state('saved_PiNetworks_' + player + '/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            # print('model loaded')
        self.train_phase = False
        self.QValue = self.out.eval(feed_dict={self.stateInput: [state]})[0]
        # print('Qvalue' + self.player)
        # print(self.QValue)
        Q_test = self.QValue * action_space
        # print('Qtest' + self.player)
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
