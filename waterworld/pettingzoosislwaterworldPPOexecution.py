from pettingzoo.sisl.waterworld import waterworld
from ppo import PPO
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


BATCH = 32
GAMMA = 0.9



def run_waterworld(parallel_env):
    
    step = 0
    num_episode = 0 
    while num_episode < 100:
        observation = parallel_env.reset()
        step = 0
        actions = {}
        accumulated_reward = 0
        number_caught = 0
        max_cycles = 500
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                action = ppo[agent].choose_action(agent_observation)
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
    parallel_env.reset()
    sess = tf.Session()
    ppo = {}
    for agent in parallel_env.agents:
        name = agent
        ppo[agent] = PPO(name,sess)
        ppo[agent].restore_actor_model(agent+"/actorppomodel.ckpt")
        ppo[agent].restore_critic_model(agent+"/criticppomodel.ckpt")
    run_waterworld(parallel_env)


