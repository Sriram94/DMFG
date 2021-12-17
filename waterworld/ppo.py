import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import gym

np.random.seed(1)
tf.set_random_seed(1)


GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 212, 2
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   
    dict(name='clip', epsilon=0.2),                 
][1]


class PPO(object):

    def __init__(self, name,sess):
        self.name = name
        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')


        with tf.variable_scope(self.name + 'Value'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope(self.name + 'critic'):
                l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            pi, pi_params = self._build_anet('pi', trainable=True)
            oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
            with tf.variable_scope(self.name + 'sample_action'):
                self.sample_op = tf.squeeze(pi.sample(1), axis=0)       
            with tf.variable_scope(self.name + 'update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            with tf.variable_scope(self.name + 'loss'):
                with tf.variable_scope(self.name + 'surrogate'):
                    ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                    surr = ratio * self.tfadv
                if METHOD['name'] == 'kl_pen':
                    self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                    kl = tf.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
                else:   
                    self.aloss = -tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

            with tf.variable_scope(self.name + 'atrain'):
                self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  
                    break
            if kl < METHOD['kl_target'] / 1.5:  
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    
        else:   
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
            
        with tf.variable_scope(self.name + 'Policy'):
            self.name_scope2 = tf.get_variable_scope().name  
            with tf.variable_scope(self.name + name):
                l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu, trainable=trainable)
                mu = tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
                sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
                norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    
    def save_actor_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Actor Model saved in path: %s" % save_path)

    def save_critic_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Critic Model saved in path: %s" % save_path)
    
    def restore_actor_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Actor Model restored")
    
    def restore_critic_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Critic Model restored")

