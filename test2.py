import numpy as np
import matplotlib.pyplot as plt

from QN import QN

ydata = []
sum = 0

agent = QN()
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

        x, x_dot, theta, theta_dot = observation
        r1 = (agent.env.x_threshold - abs(x))/agent.env.x_threshold - 0.8
        r2 = (agent.env.theta_threshold_radians - abs(theta))/agent.env.theta_threshold_radians - 0.5
        reward = r1 + r2

        agent.remember([s, a, reward, observation])
        score = score + reward

        if done:
            print("Run {} - Episode finished after {} timesteps".format(i_episode,t+1))
            sum += t+1
            print("Score: ", score)
            break

    agent.train()

    if i_episode % agent.TARGET_REPLACE_ITER == 0:
        print("Updating Weights")
        agent.update_weights()
        agent.saver.save(agent.sess, "save/model.ckpt")
        print("Model saved")

    if i_episode % 100 == 0:
        ydata.append(sum/100)
        sum = 0
        plt.close()
        plt.plot(range(0, int(i_episode/100) + 1),ydata)
        plt.ylabel('Steps')
        plt.xlabel('Times')
        plt.show(block=False)
