from pettingzoo.magent import tiger_deer_v3
from RL_brainDQN import DeepQNetwork
import csv
import numpy as np 
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


random.seed(1)

def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation



def run_tigerdeer(parallel_env):
    
    step = 0
    with open('pettingzoomagentDQN.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DQN)", "sumofrewards(DQN)"))
    deer_n_actions = 5
    tiger_n_actions = 9
    num_episode = 0 
    while num_episode < 2000:
        observation = parallel_env.reset()
        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            number_alive = [0,0]
            action_list = []
            for agent in parallel_env.agents:
                if "deer" in agent: 
                    action = random.randint(0, (deer_n_actions-1))
                else:
                    agent_observation = observation[agent]        
                    agent_observation = change_observation(agent_observation)
                    action = RL[agent].choose_action(agent_observation)
                    action_list.append(action)
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
             
            if not parallel_env.agents:  
                break
            
            for agent in parallel_env.agents: 
                if "deer" in agent: 
                    number_alive[0] = number_alive[0] + 1
                else:
                    number_alive[1] = number_alive[1] + 1
                    accumulated_reward = accumulated_reward + rewards[agent]
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                    agent_nextobservation = new_observation[agent]
                    agent_nextobservation = change_observation(agent_nextobservation)
                    RL[agent].store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
            
            
            
            observation = new_observation
            
            print("The step we are at is", step) 
                
                
            team_size = parallel_env.team_size()

            if team_size[0] == 0:                      
                break 
            
        for agent in RL:
            RL[agent].learn()

        print("The episode is", num_episode)
        
        with open('pettingzoomagentDQN.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("The accumulated reward is", accumulated_reward)
        print("The number of agents alive in deers is", number_alive[0])
        print("The number of agents alive in tigers is", number_alive[1])
   
    for agent in RL:
        RL[agent].save_model("./"+agent+"/dqnmodel.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = tiger_deer_v3.parallel_env(minimap_mode = True)
    parallel_env.seed(1)

    RL = {}
    sess = tf.Session()
    size = len(parallel_env.agents)
    for agent in parallel_env.agents:
        if 'tiger' in agent:

            new_name = agent
            RL[agent] = DeepQNetwork(9,2349,sess, name = new_name)
    
    team_size = parallel_env.team_size()
    print("The number of agents alive in deers is", team_size[0])
    print("The number of agents alive in tigers is", team_size[1])
    sess.run(tf.global_variables_initializer()) 
    run_tigerdeer(parallel_env)


