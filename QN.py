import tensorflow as tf
import numpy as np
import gym
import random
import math

class QN(object):
    def __init__(self):
        tf.set_random_seed(1)
        np.random.seed(1)

        self.sess = tf.Session()

        # Hyper Parameters
        self.BATCH_SIZE = 32
        self.LR = 0.01                   # learning rate
        self.EPSILON = 0.9               # greedy policy
        self.GAMMA = 0.9                 # reward discount
        self.TARGET_REPLACE_ITER = 100  # target update frequency
        self.MEMORY_CAPACITY = 2000
        self.MEMORY_COUNTER = 0          # for store experience
        self.RUN_TIME = 200000
        self.env = gym.make('CartPole-v1')
        self.env = self.env.unwrapped
        self.N_STATES = 4
        self.L1_NODES = 10
        self.L2_NODES = 10
        self.N_ACTIONS = 2
        self.MEMORY = []     # initialize memory

        ########################

        self.x = tf.placeholder('float', [None, self.N_STATES])
        self.y = tf.placeholder('float', [None, self.N_ACTIONS])
        self.target = tf.placeholder('float', [None, self.N_ACTIONS])

        self.keep_rate = 0.8
        self.keep_prob = tf.placeholder(tf.float32)

        self.eval_weights = {'W_fc1':tf.Variable(tf.random_normal([self.N_STATES,self.L1_NODES])),
                    'W_fc2':tf.Variable(tf.random_normal([self.L1_NODES,self.L2_NODES])),
                    'out':tf.Variable(tf.random_normal([self.L2_NODES, self.N_ACTIONS]))}

        self.eval_biases = {'b_fc1':tf.Variable(tf.random_normal([self.L1_NODES])),
                    'b_fc2':tf.Variable(tf.random_normal([self.L2_NODES])),
                    'out':tf.Variable(tf.random_normal([self.N_ACTIONS]))}

        self.target_weights = {'W_fc1':tf.Variable(tf.random_normal([self.N_STATES,self.L1_NODES])),
                    'W_fc2':tf.Variable(tf.random_normal([self.L1_NODES,self.L2_NODES])),
                    'out':tf.Variable(tf.random_normal([self.L2_NODES, self.N_ACTIONS]))}

        self.target_biases = {'b_fc1':tf.Variable(tf.random_normal([self.L1_NODES])),
                    'b_fc2':tf.Variable(tf.random_normal([self.L2_NODES])),
                    'out':tf.Variable(tf.random_normal([self.N_ACTIONS]))}

        self.e_pred = self.DQN_eval(self.x)
        self.prediction = self.DQN_target(self.x)
        self.cost = tf.reduce_mean(tf.squared_difference(self.prediction, self.target))
        self.optimizer = tf.train.RMSPropOptimizer(self.LR).minimize(self.cost)
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def DQN_eval(self,x):
        x = tf.reshape(x, shape=[-1, self.N_STATES])

        fc1 = tf.nn.sigmoid(tf.matmul(x, self.eval_weights['W_fc1']) + self.eval_biases['b_fc1'])
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, self.eval_weights['W_fc2']) + self.eval_biases['b_fc2'])

        output = tf.matmul(fc2, self.eval_weights['out']) + self.eval_biases['out']

        return output

    def DQN_target(self,x):
        x = tf.reshape(x, shape=[-1, self.N_STATES])

        fc1 = tf.nn.sigmoid(tf.matmul(x, self.eval_weights['W_fc1']) + self.eval_biases['b_fc1'])
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, self.eval_weights['W_fc2']) + self.eval_biases['b_fc2'])

        output = tf.matmul(fc2, self.eval_weights['out']) + self.eval_biases['out']

        return output

    def update_weights(self):
        copy = []
        i = 0
        for layer,_ in self.eval_weights.items():
            copy.append(self.eval_weights[layer].assign(self.target_weights[layer]))

        for layer,_ in self.eval_biases.items():
            copy.append(self.eval_biases[layer].assign(self.target_biases[layer]))

        for c in range(len(copy)):
            self.sess.run(copy[c])

    def choose_action(self,s):
        state = [np.array([s]).flatten()]
        if np.random.uniform() <= self.EPSILON:
            actions_value = self.sess.run(self.e_pred,feed_dict={self.x: state})
            action = np.argmax(actions_value[0])
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def train(self):
        for i in range(self.BATCH_SIZE):
            MEM = random.choice(self.MEMORY)
            s1 = [np.array([MEM[0]]).flatten()]
            s2 = [np.array([MEM[3]]).flatten()]

            new_target = self.sess.run(self.e_pred,feed_dict={self.x: s1})
            Qvals = self.sess.run(self.e_pred,feed_dict={self.x: s2})

            Rmax = MEM[2] + self.GAMMA * np.argmax(Qvals[0])
            new_target[0][MEM[1]] = Rmax
            self.sess.run(self.optimizer,feed_dict={self.x: s1, self.target: new_target, self.keep_prob: 0.8})

    def remember(self, mem):
        if len(self.MEMORY) < self.MEMORY_CAPACITY:
            self.MEMORY.append(mem)
        else:
            self.MEMORY[self.MEMORY_COUNTER] = mem
            if self.MEMORY_COUNTER < self.MEMORY_CAPACITY - 2:
                self.MEMORY_CAPACITY = self.MEMORY_CAPACITY + 1
            else:
                self.MEMORY_CAPACITY = 0