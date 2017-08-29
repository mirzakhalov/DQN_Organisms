import tensorflow as tf
import numpy as np
import gym
import random
import math

class DQN(object):
    def __init__(self):
        tf.set_random_seed(1)
        np.random.seed(1)

        # Hyper Parameters
        self.BATCH_SIZE = 32
        self.LR = 1e-4                   # learning rate
        self.EPSILON = 0.8               # greedy policy
        self.GAMMA = 0.995                 # reward discount
        self.TARGET_REPLACE_ITER = 5  # target update frequency
        self.MEMORY_CAPACITY = 128
        self.MEMORY_COUNTER = 0          # for store experience
        self.RUN_TIME = 200000
        self.env = gym.make('Breakout-v0')
        self.N_ACTIONS = 4
        self.MEMORY = []     # initialize memory

        ########################

        self.x = tf.placeholder('float', [None, 210*160*3])
        self.y = tf.placeholder('float', [None, 4])
        self.target = tf.placeholder('float', [None, 4])

        self.keep_rate = 0.8
        self.keep_prob = tf.placeholder(tf.float32)

        self.eval_weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
                    'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                    'W_conv3':tf.Variable(tf.random_normal([5,5,64,64])),
                    'W_conv4':tf.Variable(tf.random_normal([5,5,64,64])),
                    'W_fc':tf.Variable(tf.random_normal([14*10*64,1024])),
                    'out':tf.Variable(tf.random_normal([1024, self.N_ACTIONS]))}

        self.eval_biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                    'b_conv2':tf.Variable(tf.random_normal([64])),
                    'b_conv3':tf.Variable(tf.random_normal([64])),
                    'b_conv4':tf.Variable(tf.random_normal([64])),
                    'b_fc':tf.Variable(tf.random_normal([1024])),
                    'out':tf.Variable(tf.random_normal([self.N_ACTIONS]))}

        self.target_weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
                    'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                    'W_conv3':tf.Variable(tf.random_normal([5,5,64,64])),
                    'W_conv4':tf.Variable(tf.random_normal([5,5,64,64])),
                    'W_fc':tf.Variable(tf.random_normal([14*10*64,1024])),
                    'out':tf.Variable(tf.random_normal([1024, self.N_ACTIONS]))}

        self.target_biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                    'b_conv2':tf.Variable(tf.random_normal([64])),
                    'b_conv3':tf.Variable(tf.random_normal([64])),
                    'b_conv4':tf.Variable(tf.random_normal([64])),
                    'b_fc':tf.Variable(tf.random_normal([1024])),
                    'out':tf.Variable(tf.random_normal([self.N_ACTIONS]))}
        
        self.e_pred = self.DQN_eval(self.x)
        self.prediction = self.DQN_target(self.x)
        self.cost = tf.reduce_mean(tf.squared_difference(self.prediction, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.LR).minimize(self.cost)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def maxpool2d(self,x):
        #                        size of window         movement of window
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def DQN_eval(self,x):
        x = tf.reshape(x, shape=[-1, 210, 160, 3])

        conv1 = tf.nn.relu(self.conv2d(x, self.eval_weights['W_conv1']) + self.eval_biases['b_conv1'])
        conv1 = self.maxpool2d(conv1)
        
        conv2 = tf.nn.relu(self.conv2d(conv1, self.eval_weights['W_conv2']) + self.eval_biases['b_conv2'])
        conv2 = self.maxpool2d(conv2)

        conv3 = tf.nn.relu(self.conv2d(conv2, self.eval_weights['W_conv3']) + self.eval_biases['b_conv3'])
        conv3 = self.maxpool2d(conv3)

        conv4 = tf.nn.relu(self.conv2d(conv3, self.eval_weights['W_conv4']) + self.eval_biases['b_conv4'])
        conv4 = self.maxpool2d(conv4)

        fc = tf.reshape(conv4,[-1,14*10*64])
        fc = tf.nn.sigmoid(tf.matmul(fc, self.eval_weights['W_fc']) + self.eval_biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, self.eval_weights['out']) + self.eval_biases['out']

        return output

    def DQN_target(self,x):
        x = tf.reshape(x, shape=[-1, 210, 160, 3])

        conv1 = tf.nn.relu(self.conv2d(x, self.target_weights['W_conv1']) + self.target_biases['b_conv1'])
        conv1 = self.maxpool2d(conv1)
        
        conv2 = tf.nn.relu(self.conv2d(conv1, self.target_weights['W_conv2']) + self.target_biases['b_conv2'])
        conv2 = self.maxpool2d(conv2)

        conv3 = tf.nn.relu(self.conv2d(conv2, self.target_weights['W_conv3']) + self.target_biases['b_conv3'])
        conv3 = self.maxpool2d(conv3)

        conv4 = tf.nn.relu(self.conv2d(conv3, self.target_weights['W_conv4']) + self.target_biases['b_conv4'])
        conv4 = self.maxpool2d(conv4)

        fc = tf.reshape(conv4,[-1,14*10*64])
        fc = tf.nn.sigmoid(tf.matmul(fc, self.target_weights['W_fc']) + self.target_biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)

        output = tf.matmul(fc, self.target_weights['out']) + self.target_biases['out']

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
            actions_value = sess.run(self.e_pred,feed_dict={self.x: state})
            action = np.argmax(actions_value[0])
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def train(self):
        for i in range(self.BATCH_SIZE):
            MEM = random.choice(self.MEMORY)
            s1 = [np.array([MEM[0]]).flatten()]
            s2 = [np.array([MEM[3]]).flatten()]

            new_target = sess.run(self.e_pred,feed_dict={self.x: s1})
            Qvals = sess.run(self.e_pred,feed_dict={self.x: s2})

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

if __name__ == '__main__':
    agent = DQN()
    status = input("#: ")
    if(status == "load"):
        agent.saver.restore(agent.sess, "save/model.ckpt")
        print("Model restored")

    for i_episode in range(agent.RUN_TIME):
        observation = agent.env.reset()
        t = 0
        score = 0
        while(1):
            t = t + 1
            agent.env.render()
            s = observation
            a = agent.choose_action(s)
            observation, reward, done, info = agent.env.step(a)
            if reward == 1:
                reward = 100
            reward = reward + math.log10(t)/10

            agent.remember([s, a, reward, observation])

            score = score + reward

            if done:
                print("Run {} - Episode finished after {} timesteps".format(i_episode,t+1))
                print("Score: ", score)
                break
        agent.train()
        if i_episode % agent.TARGET_REPLACE_ITER == 0:
            print("Updating Weights")
            agent.update_weights()
            agent.saver.save(agent.sess, "save/model.ckpt")
            print("Model saved")


