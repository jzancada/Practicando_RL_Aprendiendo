import numpy as np

class Env():
    def __init__(self):
        self.N_Q = 20
        self.N_T = 24
        self.cost = np.ones(24)
        self.cost[10]=20
        self.q_space=np.linspace(0,self.N_Q-1,self.N_Q)
        self.a_space=np.array([-1,0,1])
        self.t_space = np.linspace(0,23,24)
        self.obs = self.reset()
        self.t = 0
        self.q = 0
        self.info = []
        self.final_step = 0

    def set_final_step(self, t_0):
        self.final_step = 24*30 + t_0

    def cat_t(self, t):
        # caterogire t
        t = t % 24
        t_index = np.digitize(t, self.t_space, right = True)
        self.t_cat = np.zeros(self.t_space.shape)
        self.t_cat[t_index]=1
        return self.t_cat

    def cat_q(self, q):
        # caterogire t
        q_index = np.digitize(q, self.q_space, right = True)
        self.q_cat = np.zeros(self.q_space.shape)
        self.q_cat[q_index]=1
        return self.q_cat

    def to_cat(self):
        # Q - T
        t_cat = self.cat_t(self.t)
        q_cat = self.cat_q(self.q)
        qt_cat = np.zeros(self.N_Q+self.N_T)
        qt_cat[0:self.N_Q]    = q_cat
        qt_cat[self.N_Q : self.N_Q+self.N_T] = t_cat
        return qt_cat

    def reset(self):
        self.t_reset_prev = 0
        self.q_reset_prev = 0

        self.t = np.random.choice(self.t_space)
        self.q = np.random.choice(self.q_space)

        self.t = self.t_reset_prev
        self.q = self.q_reset_prev

        # se muestrea todo el espacio de forma secuencial
        self.t_reset_prev += 1
        if self.t_reset_prev > self.N_T:
            self.t_reset_prev = 0

            self.q_reset_prev += 1
            if self.q_reset_prev > self.N_Q:
                self.q_reset_prev = 0

        self.obs = self.to_cat()
        self.set_final_step(self.t)
        return self.obs 

    # fija un estado (t,q)
    def reset_fix(self, t_0, q_0):
        self.t = t_0
        self.q = q_0
        self.obs = self.to_cat()
        self.set_final_step(self.t)
        return self.obs 

    def departure(self):
        check = False
        tq_departure = (0,0)
        if self.t in {8,9,10}:
            # tira el dado para ver si sale
            CDF_8 = .3
            CDF_9 = .6
            CDF_10 = 1
            p = np.random.rand()
            if  (self.t ==  8 and p < CDF_8) or \
                (self.t ==  9 and p < CDF_9) or \
                (self.t == 10):
                check = True
                t = np.random.choice([14, 15, 16])
                q = np.random.normal(loc = 10, scale =3)
                q = np.floor(q)
                if (q<0) or (q>self.N_Q-1):
                    q = 10
                tq_departure = (t,q)
        return tq_departure, check

    def step(self, action):
        penalizacion_q = 0
        done = False

        q_action = self.a_space[action]
        t_ = self.t +1

        # la accion es continua
        q_ = self.q + q_action

        if (q_ > self.N_Q-1):
            penalizacion_q = -1
            q_ = self.N_Q-1

        if (q_ < 0):
            penalizacion_q = -1
            q_ = 0

        delta_q = q_ - self.q
        reward = -self.cost[ int(self.t) % 24 ]*delta_q + penalizacion_q

        # penalizao si en t=9 no esta al 10
        if (t_ % 24) == 9:
            reward_en_9 = np.abs(q_)
            reward += reward_en_9*10

        # se actualizan 
        self.t = t_
        self.q = q_
        self.obs = self.to_cat()

        # sale de viaje ? #####
        tq_departure, check = self.departure ()
        if check:
            (self.t, self.q) = tq_departure
            self.obs = self.to_cat()
            # reset reward
            reward = 0 
        ######
        # batch
        if t_ > self.final_step:
            done = True
        return  self.obs, reward, done, self.info

