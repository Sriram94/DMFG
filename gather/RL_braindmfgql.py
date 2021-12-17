import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

class DMFGQL:
    def __init__(
            self,
            n_actions,
            n_features,
            sess,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=300,
            memory_size=200000,
            batch_size=64,
            name = 'none',
            e_greedy_increment=0.001,
            output_graph=False,
            model_path=False,
            execution = False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.79 if e_greedy_increment is not None else self.epsilon_max
        self.temperature = 0.1
        self.name = name
        self.learn_step_counter = 0
        self.execution = execution 
        self.memory = np.zeros((self.memory_size, n_features * 2 + self.n_actions * 2 + 2))

        self._build_net()
        self._build_net2()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []



    def copy_network(self, s):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, s)
        print("New model copied")

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s_1')  
        self.mean = tf.placeholder(tf.float32, [None, self.n_actions], name='a_1')  
        self.newmean = tf.placeholder(tf.float32, [None, self.n_actions], name='new_a1')  
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target_1')  
        
        with tf.variable_scope(self.name + 'Value'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope( self.name + 'eval_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope(self.name + 'l_1'):
                    w1 = tf.get_variable('w_1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope(self.name + 'l_3'):
                    w3 = tf.get_variable('w_3', [self.n_actions, n_l1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b_3', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l3 = tf.nn.relu(tf.matmul(self.mean, w3) + b3)
                concat_layer = tf.concat([l1, l3], axis=1)
                
                
                with tf.variable_scope(self.name + 'l_h1'):
                    wh1 = tf.get_variable('w_h1', [n_l1*2, n_l1*2], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('b_h1', [1, n_l1*2], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(concat_layer, wh1) + bh1)

                with tf.variable_scope(self.name + 'l_h2'):
                    wh2 = tf.get_variable('w_h2', [n_l1*2, n_l1*2], initializer=w_initializer, collections=c_names)
                    bh2 = tf.get_variable('b_h2', [1, n_l1*2], initializer=b_initializer, collections=c_names)
                    lh2 = tf.nn.relu(tf.matmul(lh1, wh2) + bh2)
                
                
                
                
                
                
                with tf.variable_scope(self.name + 'l_2'):
                    w2 = tf.get_variable('w_2', [n_l1*2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(lh2, w2) + b2

            with tf.variable_scope(self.name + 'loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope(self.name + 'train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                
            with tf.variable_scope(self.name + 'predict'):
                self.predict = tf.nn.softmax(self.q_eval / self.temperature)
                        
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_1_')    
            with tf.variable_scope('target_net_1'):
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope(self.name + 'l_1'):
                    w1 = tf.get_variable('w_1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
                
                with tf.variable_scope(self.name + 'l_3'):
                    w3 = tf.get_variable('w_3', [self.n_actions, n_l1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b_3', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l3 = tf.nn.relu(tf.matmul(self.newmean, w3) + b3)
                
                concat_layer = tf.concat([l1, l3], axis=1)



                with tf.variable_scope(self.name + 'l_h1'):
                    wh1 = tf.get_variable('w_h1', [n_l1*2, n_l1*2], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('b_h1', [1, n_l1*2], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(concat_layer, wh1) + bh1)

                with tf.variable_scope(self.name + 'l_h2'):
                    wh2 = tf.get_variable('w_h2', [n_l1*2, n_l1*2], initializer=w_initializer, collections=c_names)
                    bh2 = tf.get_variable('b_h2', [1, n_l1*2], initializer=b_initializer, collections=c_names)
                    lh2 = tf.nn.relu(tf.matmul(lh1, wh2) + bh2)





                with tf.variable_scope(self.name + 'l_2'):
                    w2 = tf.get_variable('w_2', [n_l1*2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.matmul(lh2, w2) + b2

    def _build_net2(self):

        self.mean_action = tf.placeholder(tf.float32, [None, self.n_actions], name='meanaction')  
        self.target_mean = tf.placeholder(tf.float32, [None, self.n_actions], name='target_mean')  
        self.s_new = tf.placeholder(tf.float32, [None, self.n_features], name='s_2')  

        with tf.variable_scope(self.name + 'meanfieldnetwork'):
            self.name_scope2 = tf.get_variable_scope().name
            
            with tf.variable_scope(self.name + 'meanaction_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['meanfield_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope(self.name + 'meanactionl'):
                    w1 = tf.get_variable('meanactionw', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('meanactionb', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_new, w1) + b1)
                

                with tf.variable_scope(self.name + 'meanactionl_3'):
                    w3 = tf.get_variable('meanactionw_3', [self.n_actions, n_l1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('meanactionb_3', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l3 = tf.nn.relu(tf.matmul(self.mean_action, w3) + b3)
                
                concat_layer = tf.concat([l1, l3], axis=1)



                with tf.variable_scope(self.name + 'meanactionl_h1'):
                    wh1 = tf.get_variable('meanactionw_h1', [n_l1*2, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('meanactionb_h1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(concat_layer, wh1) + bh1)

                with tf.variable_scope(self.name + 'meanactionl_h2'):
                    wh2 = tf.get_variable('meanactionw_h2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh2 = tf.get_variable('meanactionb_h2', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh2 = tf.nn.relu(tf.matmul(lh1, wh2) + bh2)






                with tf.variable_scope(self.name + 'meanactionl2'):
                    w2 = tf.get_variable('meanactionw2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('meanactionb2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.new_mean = tf.nn.softmax(tf.matmul(lh2, w2) + b2)


            with tf.variable_scope(self.name + 'meanactionloss2'):
                self.loss2 = tf.reduce_mean(tf.squared_difference(self.target_mean, self.new_mean))
            with tf.variable_scope(self.name + 'meanactiontrain2'):
                self._train_op2 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss2)    
    
    
    
    
    
    
    
    def store_transition(self, s, a, a1, new_a1, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        a1 = np.array(a1)
        a1 = a1[0]
        new_a1 = np.array(new_a1)
        new_a1 = new_a1[0]
        transition = np.hstack((s, a1, new_a1, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, a1):
        
        a1 = np.array(a1)
        observation = np.expand_dims(observation, axis=0) 
        if np.random.uniform() <= self.epsilon:
            actions_value = self.sess.run(self.predict, feed_dict={self.s: observation, self.mean: a1})
            if self.execution == False:
                action = np.random.choice(np.arange(actions_value.shape[1]), p=actions_value.ravel())
            else:
                action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action





    def get_mean_action(self, observation, a1):
        
        
        a1 = np.array(a1)
        observation = np.expand_dims(observation, axis=0) 
        mean_value = self.sess.run(self.new_mean, feed_dict={self.s_new: observation, self.mean_action: a1})
        mean_value = mean_value[0]
        return mean_value


    def learn_mean_action(self, observation, a1, previous_mean):
        
        a1 = np.array(a1)
        previous_mean = np.array(previous_mean)
        observation = np.expand_dims(observation, axis=0) 
        _, self.cost2 = self.sess.run([self._train_op2, self.loss2], feed_dict={self.s_new: observation, self.mean_action: previous_mean, self.target_mean:a1})
        return self.cost2





    def learn(self):

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  
                self.s: batch_memory[:, :self.n_features],  
                self.mean: batch_memory[:, self.n_features:self.n_features+self.n_actions],
                self.newmean: batch_memory[:, self.n_features+self.n_actions:self.n_features+self.n_actions+self.n_actions]
                })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+self.n_actions+self.n_actions].astype(int)
        reward = batch_memory[:, self.n_features+self.n_actions+self.n_actions+1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.mean: batch_memory[:, self.n_features:self.n_features+self.n_actions],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self.sess.run(self.replace_target_op)


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



    def save_model_meanfield(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model_meanfield(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")





    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



