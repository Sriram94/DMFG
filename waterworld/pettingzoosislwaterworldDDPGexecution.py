from pettingzoo.sisl.waterworld import waterworld
from ddpg import Actor
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

    num_episode = 0
    actions = {}
    while num_episode < 100:
        observation = parallel_env.reset()        
        
        
        accumulated_reward = 0
        number_caught = 0
        max_cycles = 500 
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                action = actor[agent].choose_action(agent_observation)
                action = np.clip(action, -action_bound, action_bound) 
                actions[agent] = action
            new_observation, rewards, dones, infos = parallel_env.step(actions)
            
            number_caught = number_caught + parallel_env.return_number_caught()

            if not parallel_env.agents:  
                break

            for agent in parallel_env.agents:
                accumulated_reward = accumulated_reward + rewards[agent]

            
            
             
            observation = new_observation 
            print("The step we are at is", step)


        print("the total food captured is", number_caught)


        num_episode = num_episode + 1   
        print("The episode is", num_episode)




    print("The episode is", num_episode)
    print('game over')


if __name__ == "__main__":
    parallel_env = waterworld.parallel_env(n_pursuers=25, n_evaders=25, encounter_reward=1)
    parallel_env.reset()
    actor = {}
    sess = tf.Session()
    for agent in parallel_env.agents:
        name = agent 
        actor[agent] = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT, name)
        actor[agent].restore_model(agent+"/ddpgactormodel.ckpt")

    
    sess.run(tf.global_variables_initializer())
    run_waterworld(parallel_env)
