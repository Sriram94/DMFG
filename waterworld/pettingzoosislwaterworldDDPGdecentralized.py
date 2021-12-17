from pettingzoo.sisl.waterworld import waterworld
from ddpg import Actor
from ddpg import Critic
from ddpg import Memory
import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

LR_A = 0.001    
LR_C = 0.002    
GAMMA = 0.9     
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

state_dim = 212
action_dim = 2
action_bound = 1


def run_waterworld(parallel_env):

    with open('pettingzoosislwaterworldDDPG.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DDPG)"))
    num_episode = 0
    actions = {}
    var = 3 
    while num_episode < 2000:
        observation = parallel_env.reset()        
        
        
        accumulated_reward = 0
        number_caught = 0
        max_cycles = 500 
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                action = actor[agent].choose_action(agent_observation)
                action = np.random.normal(action, var)
                action = np.clip(action, -action_bound, action_bound) 
                actions[agent] = action
            new_observation, rewards, dones, infos = parallel_env.step(actions)
            
            number_caught = number_caught + parallel_env.return_number_caught()

            if not parallel_env.agents:  
                break

            for agent in parallel_env.agents:
                accumulated_reward = accumulated_reward + rewards[agent]

                M[agent].store_transition(observation[agent], actions[agent], rewards[agent], new_observation[agent])
            
                if M[agent].pointer > MEMORY_CAPACITY:
                    var *= .9995    
                    b_M = M[agent].sample(BATCH_SIZE)
                    b_s = b_M[:, :state_dim]
                    b_a = b_M[:, state_dim: state_dim + action_dim]
                    b_r = b_M[:, -state_dim - 1: -state_dim]
                    b_s_ = b_M[:, -state_dim:]

                    critic[agent].learn(b_s, b_a, b_r, b_s_)
                    actor[agent].learn(b_s)
            
             
            observation = new_observation 
            print("The step we are at is", step)


        #print("the total food captured is", number_caught)


        with open('pettingzoosislwaterworldDDPG.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1   
        print("The episode is", num_episode)



    for agent in parallel_env.agents:

        actor[agent].save_model(agent+"/ddpgactormodel.ckpt")
        critic[agent].save_model(agent+"/ddpgcriticmodel.ckpt")


    print("The episode is", num_episode)
    print('game over')


if __name__ == "__main__":
    parallel_env = waterworld.parallel_env(n_pursuers=25, n_evaders=25, encounter_reward=1)
    parallel_env.seed(1)
    parallel_env.reset()
    actor = {}
    critic = {} 
    M = {}
    sess = tf.Session()
    for agent in parallel_env.agents:
        name = agent 
        actor[agent] = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT, name)
        critic[agent] = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor[agent].a, actor[agent].a_, name)
        
    for agent in parallel_env.agents:
        actor[agent].add_grad_to_graph(critic[agent].a_grads)
    
    sess.run(tf.global_variables_initializer())
    
    for agent in parallel_env.agents: 
        M[agent] = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
    
    run_waterworld(parallel_env)

