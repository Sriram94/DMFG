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
    with open('pettingzoosislwaterworldPPO.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(PPO)"))
    num_episode = 0 
    while num_episode < 2000:
        observation = parallel_env.reset()
        buffer_s = {}
        buffer_a = {}
        buffer_r = {}
        for agent in parallel_env.agents: 
            buffer_s[agent] = []
            buffer_a[agent] = []
            buffer_r[agent] = []
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
                buffer_s[agent].append(observation[agent])
                buffer_a[agent].append(actions[agent])
                buffer_r[agent].append(rewards[agent])
            
                if (step+1) % BATCH == 0 or (step+1) == max_cycles:
                    v_s_ = ppo[agent].get_v(observation[agent])
                    discounted_r = []
                    for r in buffer_r[agent][::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s[agent]), np.vstack(buffer_a[agent]), np.array(discounted_r)[:, np.newaxis]
                    buffer_s[agent] = []
                    buffer_a[agent] = []
                    buffer_r[agent] = []
                    ppo[agent].update(bs, ba, br)
            
        
            observation = new_observation
            print("The step we are at is", step)


        with open('pettingzoosislwaterworldPPO.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))

        #print("the total food captured is", number_caught)
        num_episode = num_episode + 1            
        print("The episode is", num_episode)
    
    for agent in ppo:
        ppo[agent].save_actor_model(agent+"/actorppomodel.ckpt")
        ppo[agent].save_critic_model(agent+"/criticppomodel.ckpt")
    
    
    print('game over')


if __name__ == "__main__":
    parallel_env = waterworld.parallel_env(n_pursuers=25, n_evaders=25, encounter_reward=1)
    parallel_env.seed(1)
    parallel_env.reset()
    sess = tf.Session()
    ppo = {}
    for agent in parallel_env.agents:
        name = agent
        ppo[agent] = PPO(name,sess)
    run_waterworld(parallel_env)


