import numpy as np

class Env():
    # cada 15 minutos
    def __init__(self):
        self.Q = 1 # p.u.
        self.N_T = 14
        self.q_space=np.linspace(0,self.Q)
        self.a_space=np.array([-3,-1,0,1,3])
        self.t_space = np.linspace(0,23,24)
        self.t_final_epoch = 0
        self.precios_red = np.array([-14, -13, -12, -11, 0, 0, 80, 79, 78, 77, 76, 75, 74, -15], dtype='float')
        self.precios_red *= .5
        self.obs = self.reset()

    def set_obs(self, t_0, q_0):
        x = np.zeros((11+4,))
        t_0_i = int(t_0)
        x[ 0] = q_0
        x[ 1] = self.precios_red[ int(t_0    ) % self.N_T ]
        x[ 2] = self.precios_red[ int(t_0 + 1) % self.N_T ]
        x[ 3] = self.precios_red[ int(t_0 + 2) % self.N_T ]
        x[ 4] = self.precios_red[ int(t_0 + 3) % self.N_T ]
        x[ 5] = self.precios_red[ int(t_0 + 4) % self.N_T ]
        x[ 6] = self.precios_red[ int(t_0 + 5) % self.N_T ]
        x[ 7] = self.precios_red[ int(t_0 + 6) % self.N_T ]
        x[ 8] = self.precios_red[ int(t_0 + 7) % self.N_T ]
        x[ 9] = self.precios_red[ int(t_0 + 8) % self.N_T ]
        x[10] = self.precios_red[ int(t_0 + 9) % self.N_T ]
        # q_req
        q_req = 0.9
        if  (t_0_i % self.N_T == 4) or \
            (t_0_i % self.N_T == 5) or \
            (t_0_i % self.N_T == 6) or \
            (t_0_i % self.N_T == 7)  :
            q_req = 0.9
        x[11+0] = q_req
        x[11+1] = q_req
        x[11+2] = q_req
        x[11+3] = q_req

        self.obs = x
        return self.obs

    def reset(self):
        # se coge un dia aleatorio
        self.t = int(np.random.uniform(0,self.N_T))
        self.q = np.random.choice(self.q_space) 
        # se fuerza a cero
        self.t = 0
        self.q = 0
        # self.q = 0 ################
        self.t_final_epoch =self.N_T*60 + self.t
        info={}
        return self.set_obs(self.t, self.q) , info

    # fija un estado (q)
    def reset_fix_q_0(self, q_0):
        self.t = 0
        self.q = q_0
        self.obs = self.set_obs(self.t, self.q)
        info={}
        return self.obs, info

    # fija un estado (q)
    def reset_fix_t_q_0(self, t_0, q_0):
        self.reset_fix_q_0 (q_0)
        self.t = t_0 # ojo que reset_fix_q_0 pisa
        self.obs = self.set_obs(self.t, self.q)
        return self.obs 

    def step(self, action_v, info):
        penalizacion_q_0 = 0
        penalizacion_q_1 = 0
        done = False

        action_0 = action_v[0] * (1/4) # 
        action_1 = action_v[1] * (1/4) # 

        # q_action = self.a_space[action]
        q_action_0 = action_0
        q_action_1 = action_1

        t_ = self.t + 1 

        # la accion es continua
        q_0 = self.q + q_action_0
        q_1 = self.q + q_action_0 + q_action_1

        # penalizo q_0
        if (q_0 > self.Q):
            extra_q_0 = q_0 -  self.Q
            penalizacion_q_0 = -5*extra_q_0 
            q_0 = self.Q

        if (q_0 < 0):
            extra_q_0 = -q_0
            penalizacion_q_0 = -5*extra_q_0
            q_0 = 0

        # penalizo q_1
        if (q_1 > self.Q):
            extra_q_1 = q_1 -  self.Q
            penalizacion_q_1 = -5*extra_q_1
            q_1 = self.Q

        if (q_1 < 0):
            extra_q_1 = -q_1
            penalizacion_q_1 = -5*extra_q_1
            q_1 = 0

        delta_q_0 = q_0  - self.q
        delta_q_1 = q_1  - q_0

        # precio
        reward =    -self.precios_red[ int(self.t    ) % self.N_T ] * delta_q_0 + \
                    -self.precios_red[ int(self.t + 1) % self.N_T ] * delta_q_1 * 0 + \
            penalizacion_q_0 +  penalizacion_q_1

        # reward req
        if info['epoch'] > 50:
            q_est_1 = self.obs[0] + action_0
            q_est_2 = q_est_1 + action_1

            q_req_1 = self.obs[11]
            q_req_2 = self.obs[12]

            delta = max(0, q_req_1 - q_est_1)
            reward -= delta*5
            delta = max(0, q_req_2 - q_est_2)
            reward -= delta*5
        else:
            pass
        
        # consumo bateria. para todas las acciones excepto para accion ==2
        reward -= np.abs(delta_q_0)*2*0
        # fin consumo bateria

        # se actualizan 
        self.t = t_
        self.q = q_0
        self.obs = self.set_obs(self.t, self.q)

        # batch
        if t_ > self.t_final_epoch:
            done = True

        return  self.obs, reward, done, info

