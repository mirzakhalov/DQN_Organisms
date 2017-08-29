import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 10   # pixels
ENV_H = 50  # grid height
ENV_W = 50 # grid width


class Enviorment(tk.Tk, object):
    def __init__(self):
        super(Enviorment, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r','s']
        self.n_actions = len(self.action_space)
        self.title('Enviorment')
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='blue',
                           height=ENV_H * UNIT,
                           width=ENV_W * UNIT)

        # create grids
        for c in range(0, ENV_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, ENV_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, ENV_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, ENV_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - (UNIT * 3 / 8), hell1_center[1] - (UNIT * 3 / 8),
            hell1_center[0] + (UNIT * 3 / 8), hell1_center[1] + (UNIT * 3 / 8),
            fill='red')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - (UNIT * 3 / 8), hell2_center[1] - (UNIT * 3 / 8),
            hell2_center[0] + (UNIT * 3 / 8), hell2_center[1] + (UNIT * 3 / 8),
            fill='red')

        # create food oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - (UNIT * 3 / 8), oval_center[1] - (UNIT * 3 / 8),
            oval_center[0] + (UNIT * 3 / 8), oval_center[1] + (UNIT * 3 / 8),
            fill='yellow')

        # create agent rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - (UNIT * 3 / 8), origin[1] - (UNIT * 3 / 8),
            origin[0] + (UNIT * 3 / 8), origin[1] + (UNIT * 3 / 8),
            fill='black')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT/2, UNIT/2])
        self.rect = self.canvas.create_rectangle(
            origin[0] - (UNIT * 3 / 8), origin[1] - (UNIT * 3 / 8),
            origin[0] + (UNIT * 3 / 8), origin[1] + (UNIT * 3 / 8),
            fill='black')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (ENV_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (ENV_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            #print(s,"\n",r,"\n",done,"\n")
            if done:
                break

if __name__ == '__main__':
    env = Enviorment()
    env.after(100, update)
    env.mainloop()