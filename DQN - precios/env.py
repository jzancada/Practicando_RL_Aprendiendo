import numpy as np
from precios_red import Precios_red

class Env():
    def __init__(self):
        self.N_Q = 13
        self.N_T = 24
        self.q_space=np.linspace(0,self.N_Q-1,self.N_Q)
        self.a_space=np.array([-3,-1,0,1,3])
        self.t_space = np.linspace(0,23,24)
        self.t_final_epoch = 0
        self.precios_red = Precios_red()
        self.t_precios_red = 0
        self.obs = self.reset()
        self.info = []

    def set_obs(self, t_0, q_0):
        dia = np.linspace(0,23,24, dtype=int)
        x = np.zeros((25,))
        x[0] = q_0
        x[1:25] = self.precios_red.value[ (t_0+dia) ]
        return x

    def reset(self):
        # se coge un dia aleatorio
        self.t_precios_red = int(np.random.uniform(0,650))*24
        self.t = self.t_precios_red
        self.q = np.random.choice(self.q_space)
        # se fuerza a cero
        self.q = 0 ################
        self.t_final_epoch = 24*30 + self.t
        return self.set_obs(self.t, self.q)

    # fija un estado (q)
    def reset_fix_q_0(self, q_0):
        self.t = self.t_precios_red
        self.q = q_0
        self.obs = self.set_obs(self.t, self.q)
        return self.obs 

    # fija un estado (q)
    def reset_fix_t_q_0(self, t_0, q_0):
        self.reset_fix_q_0 (q_0)
        self.t = t_0 # ojo que reset_fix_q_0 pisa
        self.obs = self.set_obs(self.t, self.q)
        return self.obs 

    def step(self, action):
        penalizacion_q = 0
        done = False

        q_action = self.a_space[action]
        t_ = self.t +1

        # la accion es continua
        q_ = self.q + q_action

        if (q_ > self.N_Q-1):
            penalizacion_q = -5
            q_ = self.N_Q-1

        if (q_ < 0):
            penalizacion_q = -5
            q_ = 0

        delta_q = q_ - self.q
        reward = -self.precios_red.value[ self.t ]*delta_q + penalizacion_q

        # consumo bateria. para todas las acciones excepto para accion ==2
        reward -= np.abs(delta_q)*2
        # fin consumo bateria

        # se actualizan 
        self.t = t_
        self.q = q_
        self.obs = self.set_obs(self.t, self.q)

        # batch
        if t_ > self.t_final_epoch:
            done = True
        if False and np.abs(penalizacion_q) >0:
            done = True

        # incrementa tiempo 
        self.t_precios_red += 1           
        # incrementa tiempo 
        return  self.obs, reward, done, self.info

