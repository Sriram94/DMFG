import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(1)
tf.set_random_seed(1)

GAMMA = 0.9     


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, name = 'none'):
        self.sess = sess
        self.name = name
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  


        with tf.variable_scope(self.name + 'MFACactor'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope(self.name + 'Actor'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=256,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l1'
                )


                l2 = tf.layers.dense(
                    inputs=l1,
                    units=256,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l2'
                )


                l3 = tf.layers.dense(
                    inputs=l2,
                    units=256,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l3'
                )


                self.acts_prob = tf.layers.dense(
                    inputs=l3,
                    units=n_actions,    
                    activation=tf.nn.softmax,   
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='acts_prob'
                )

            with tf.variable_scope(self.name + 'exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  

            with tf.variable_scope(self.name + 'train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   
    
    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")


class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01, name = 'none'):
        self.sess = sess
        self.name = name
        self.n_actions = n_actions
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.mean = tf.placeholder(tf.float32, [1, self.n_actions], "meanaction")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        with tf.variable_scope(self.name + 'MFACcritic'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope(self.name + 'Critic'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=256,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l1'
                )

                l2 = tf.layers.dense(
                    inputs=self.mean,
                    units=256,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l2'
                )
                
                concat_layer = tf.concat([l1, l2], axis=1)


                l3 = tf.layers.dense(
                    inputs=concat_layer,
                    units=256,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l3'
                )




                l4 = tf.layers.dense(
                    inputs=l3,
                    units=256,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l4'
                )



                self.v = tf.layers.dense(
                    inputs=l4,
                    units=1,  
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='V'
                )

            with tf.variable_scope(self.name + 'squared_TD_error'):
                self.td_error = self.r + GAMMA * self.v_ - self.v
                self.loss = tf.square(self.td_error)    
            with tf.variable_scope(self.name + 'train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, a1, r, s_):
        
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        a1 = np.array(a1)
        v_ = self.sess.run(self.v, {self.s: s_, self.mean: a1})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                {self.s: s, self.mean: a1, self.v_: v_, self.r: r})
        return td_error

    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")
