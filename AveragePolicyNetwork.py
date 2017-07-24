import numpy as np
import tensorflow as tf
import random

class Pi:
    """class for average-policy network"""
    def __init__(self, ACTION_NUM, STATE_NUM, SLMemory):
        self.ACTION_NUM = ACTION_NUM
        self.STATE_NUM = STATE_NUM
        self.SLMemory = SLMemory
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
        h_layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_layer1, W2), b2))
        self.output = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_layer2, W3), b3))
        self.cost = tf.reduce_mean(-tf.log(tf.multiply(self.output, self.actionOutput)))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_SLNetworks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def trainPiNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.SLMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]

        self.trainStep.run(feed_dict={
            self.actionOutput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 100 == 0:
            saver.save(self.session, 'saved_SLNetworks/' + 'network' + '-SL', global_step=self.timeStep)

        self.timeStep += 1

    def getAction(self, action_space, state):
        QValue = self.output.eval(feed_dict={self.stateInput: [state]})[0]
        action_index = np.argmax(QValue * action_space)
        return action_index
