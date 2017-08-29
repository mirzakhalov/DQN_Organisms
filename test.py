from Enviorment import Enviorment

env = Enviorment()
s = env.reset()
while True:
    env.render()
    s, r, done = env.step(int(input("action :")))