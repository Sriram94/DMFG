import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(1)
tf.set_random_seed(1)

GAMMA = 0.9     


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.001, name='none'):
        self.sess = sess
        self.name = name
        self.s = tf.placeholder(tf.float32, [1, n_features], "DMFGac_state")
        self.a = tf.placeholder(tf.float32, None, "DMFGac_act")
        self.td_error = tf.placeholder(tf.float32, None, "DMFGac_td_error")  


        with tf.variable_scope(self.name + 'DMFGACactor'):
            self.name_scope = tf.get_variable_scope().name
            with tf.variable_scope(self.name + 'DMFGac_Actor'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=256,  
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='DMFGac_l1'
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





                mu = tf.layers.dense(
                    inputs=l3,
                    units=2,  
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='DMFGac_mu'
                )

                sigma = tf.layers.dense(
                    inputs=l3,
                    units=2,  
                    activation=tf.nn.softplus,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(1.),  
                    name='DMFGac_sigma'
                )
                
                
                global_step = tf.Variable(0, trainable=False)
                self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
                self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

                self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])
            
            
            with tf.name_scope(self.name + 'DMFGac_exp_v'):
                log_prob = self.normal_dist.log_prob(self.a)  
                self.exp_v = log_prob * self.td_error  
                self.exp_v += 0.01 * self.normal_dist.entropy()

            with tf.name_scope(self.name + 'DMFGac_train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    

    
    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.sess.run(self.action, {self.s: s}) 
        action = action[0]
        return action


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
    def __init__(self, sess, n_features, n_meanaction, lr=0.01, name = 'none'):
        self.sess = sess
        self.name = name
        self.n_meanaction = n_meanaction
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.mean = tf.placeholder(tf.float32, [1, self.n_meanaction], "meanaction")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        with tf.variable_scope(self.name + 'DMFGACcritic'):
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

    def learn(self, s, a1, new_a1, r, s_):
        
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        a1 = np.array(a1)
        a1 = a1[np.newaxis, :]
        new_a1 = np.array(new_a1)
        new_a1 = new_a1[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_, self.mean: new_a1})
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


     
class Meanaction(object):
    def __init__(self, sess, n_features, n_meanaction, lr=0.01, name = 'none'):
        self.sess = sess
        self.name = name
        self.n_meanaction = n_meanaction
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.mean = tf.placeholder(tf.float32, [1, n_meanaction], "meanaction")
        self.target_mean = tf.placeholder(tf.float32, [1, n_meanaction], "target_mean")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")


        with tf.variable_scope(self.name + 'DMFGACmeanfield'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope(self.name + 'meanactionnetwork'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l1'
                )

                l2 = tf.layers.dense(
                    inputs=self.mean,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l2'
                )
                
                concat_layer = tf.concat([l1, l2], axis=1)


                l3 = tf.layers.dense(
                    inputs=concat_layer,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l3'
                )




                l4 = tf.layers.dense(
                    inputs=l3,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='l4'
                )



                self.new_mean = tf.layers.dense(
                    inputs=l4,
                    units=n_meanaction,  
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='meanaction'
                )

            with tf.variable_scope(self.name + 'squared_TD_error'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.target_mean, self.new_mean))
            with tf.variable_scope(self.name + 'train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn_mean_action(self, s, target_mean, previous_mean):
        
        s = s[np.newaxis, :]
        target_mean = np.array(target_mean)
        target_mean = target_mean[np.newaxis, :]
        previous_mean = np.array(previous_mean)
        previous_mean = previous_mean[np.newaxis, :]
        self.sess.run(self.train_op, {self.s: s, self.mean: previous_mean, self.target_mean: target_mean})


    def normalize(self, a):
        b = a.min()
        c = a.max() 
        if b >= -1 and c <= 1: 
            return a 
        else: 
            d = 2.*(a - np.min(a))/np.ptp(a)-1
            return d

    def get_mean_action(self, s, previous_mean): 
            
        s = s[np.newaxis, :]
        previous_mean = np.array(previous_mean)
        previous_mean = previous_mean[np.newaxis, :]
        mean_value = self.sess.run(self.new_mean, {self.s: s, self.mean: previous_mean})
        mean_value = mean_value[0]
        
        mean_value = self.normalize(mean_value)

        return mean_value
        

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
