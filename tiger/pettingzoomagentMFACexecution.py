from pettingzoo.magent import tiger_deer_v3
from RL_actorcritic import Actor
from RL_actorcritic import Critic
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random 



def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation



def run_tiger_deer(parallel_env):
    
    step = 0
    deer_n_actions = 5
    n_actions = 9
    
    
    num_episode = 0 
    while num_episode < 100:
        agent_num = 0
        observation = parallel_env.reset()
        
        list_tmp = []
        for i in range(n_actions):
            list_tmp.append(i)


        mean_action = {}
        for agent in parallel_env.agents:
            if "tiger" in agent:
                mean_action[agent] = np.mean(list(map(lambda x: np.eye(n_actions)[x], list_tmp)), axis=0, keepdims=True)
        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            number_alive = [0,0]
            for agent in parallel_env.agents:
                if "deer" in agent: 
                    action = random.randint(0, (deer_n_actions-1))
                else:
                    agent_observation = observation[agent]       
                    agent_observation = change_observation(agent_observation)
                    action = actor[agent].choose_action(agent_observation)
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            
            
            for agent in parallel_env.agents: 
                if "tiger" in agent:    
                    accumulated_reward = accumulated_reward + rewards[agent]


            


            
            
            
            observation = new_observation    
            print("The step we are at is", step)
                        

        print("The episode is", num_episode)


        num_episode = num_episode + 1            
        team_size = parallel_env.team_size()
        print("The number of deers alive is", team_size[0])
        print("The number of tigers alive is", team_size[1])
        print("The accumulated reward is", accumulated_reward)





    print('game over')


if __name__ == "__main__":
    parallel_env = tiger_deer_v3.parallel_env(minimap_mode = True)
    parallel_env.seed(1)
    sess = tf.Session()
    actor = {}
    size = len(parallel_env.agents)


    for agent in parallel_env.agents:
        if 'tiger' in agent:
            name = agent
            actor[agent] = Actor(sess, n_features=2349, n_actions=9, lr=0.00001, name = name)
            actor[agent].restore_model("./"+agent+"/actormodel.ckpt")

    sess.run(tf.global_variables_initializer())
    run_tiger_deer(parallel_env)

