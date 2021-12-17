from pettingzoo.magent import gather_v2
from RL_brainDQN import DeepQNetwork
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation



def run_gather(parallel_env):
    
    step = 0
    with open('pettingzoomagentDQN.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DQN)", "sumofrewards(DQN)"))
    
    num_episode = 0 
    while num_episode < 2000:
        observation = parallel_env.reset()
        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]        
                agent_observation = change_observation(agent_observation)
                action = RL[agent].choose_action(agent_observation)
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            for agent in parallel_env.agents: 
                accumulated_reward = accumulated_reward + rewards[agent]
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)
                RL[agent].store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
            
            
            
            observation = new_observation
            
            
                
            print("The step we are at is", step)
                
            



        for agent in parallel_env.agents:
            RL[agent].learn()        
        
        
        
        
        print("The episode is", num_episode)
        team_size = parallel_env.team_size()
        accumulated_reward = accumulated_reward/team_size[0] 
        with open('pettingzoomagentDQN.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("The amount of food remaining is", parallel_env.food_size())
        print("The accumulated reward is", accumulated_reward)
        print("The number of agents alive is", len(parallel_env.agents))

    for agent in parallel_env.agents:
        RL[agent].save_model("./"+agent+"/dqnmodel.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = gather_v2.parallel_env()
    parallel_env.seed(1)
    RL = {}
    sess = tf.Session()
    size = len(parallel_env.agents)
    for agent in parallel_env.agents:

        new_name = agent        
        
        RL[agent] = DeepQNetwork(33,9675,sess,name = new_name)

    sess.run(tf.global_variables_initializer())
    print("The number of agents is", len(parallel_env.agents))
    print("The amount of food remaining is", parallel_env.food_size())
    run_gather(parallel_env)


