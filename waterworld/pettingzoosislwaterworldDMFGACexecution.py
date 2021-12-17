from pettingzoo.sisl.waterworld import waterworld
from RL_dmfgac import Actor
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




def run_waterworld(parallel_env):
    
    n_actions = 25
    
    num_episode = 0 
    while num_episode < 100:
        observation = parallel_env.reset()


        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        number_caught = 0
        for step in range(max_cycles):
            
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                action = actor[agent].choose_action(agent_observation)
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

    
    

 
    print('game over')


if __name__ == "__main__":
    parallel_env = waterworld.parallel_env(n_pursuers=25, n_evaders=25, encounter_reward=1)
    parallel_env.seed(1)
    parallel_env.reset()

    actor = {}
    sess = tf.Session()
    size = len(parallel_env.agents)
    action_bound = [-1, 1]
    for agent in parallel_env.agents:
        name = agent

        actor[agent] = Actor(sess, n_features=212, action_bound = action_bound, lr=0.00001, name = name)
        
        actor[agent].restore_model("./"+agent+"/dmfgacactormodel.ckpt") 
    
    sess.run(tf.global_variables_initializer())
    run_waterworld(parallel_env)


