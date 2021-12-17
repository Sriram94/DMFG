from pettingzoo.magent import combined_arms_v3
from RL_dmfgac import Actor
from RL_dmfgac import Meanaction
from RL_dmfgac import Critic
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



def run_battle(parallel_env):
    
    n_actions = 25
    with open('pettingzoomagentDMFGAC.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DMFGAC)", "sumofrewards(DMFGAC)"))
    
    num_episode = 0 
    while num_episode < 2000:
        agent_num = 0
        observation = parallel_env.reset()
        mean_list = [[] for _ in range(2)]

        list_tmp = []
        for i in range(n_actions):
            list_tmp.append(i)


        estimated_mean_action = {}
        estimated_previous_mean_action = {}
        previous_observed_mean_action = {}
        observed_mean_action = {}



        for agent in parallel_env.agents:
            estimated_mean_action[agent] = np.mean(list(map(lambda x: np.eye(n_actions)[x], list_tmp)), axis=0, keepdims=True)
            observed_mean_action[agent] = np.mean(list(map(lambda x: np.eye(n_actions)[x], list_tmp)), axis=0, keepdims=True)
            estimated_mean_action[agent] = estimated_mean_action[agent][0]


        estimated_previous_mean_action = estimated_mean_action.copy()
        observed_previous_mean_action = observed_mean_action.copy()

        
        accumulated_reward = [0,0,0,0]
        max_cycles = 500
        actions = {}
        previous_mean = {}
        for step in range(max_cycles): 
            for agent in parallel_env.agents:
                
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                action = actor[agent].choose_action(agent_observation)
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            new_action = {}
            for agent in parallel_env.agents: 
                new_action[agent] = actions[agent]

            estimated_previous_mean_action = estimated_mean_action.copy()
            observed_previous_mean_action = observed_mean_action.copy()



            neighbour_action = parallel_env.get_neighbour_list(new_action, 6, n_actions)
            
            
            
            
            for key in neighbour_action:
                observed_mean_action[key] = np.mean(list(map(lambda x: np.eye(n_actions)[x], neighbour_action[key])), axis=0, keepdims=True)


            
            for agent in parallel_env.agents: 
                if "red" in agent:
                    if "mele" in agent:
                        team = 0
                    else:
                        team = 2

                elif "blue" in agent:
                    if "mele" in agent:
                        team = 1
                    else:
                        team = 3 

                accumulated_reward[team] = accumulated_reward[team] + rewards[agent]
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)
                meanactionobject[agent].learn_mean_action(agent_observation, observed_mean_action[agent], observed_previous_mean_action[agent])
                estimated_mean_action[agent] = meanactionobject[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])

                td_error = critic[agent].learn(agent_observation, estimated_previous_mean_action[agent], estimated_mean_action[agent], rewards[agent], agent_nextobservation)
                actor[agent].learn(agent_observation, actions[agent], td_error)
            
            
            observation = new_observation
            
            
                
            print("The step we are at is", step)
            

        num_episode = num_episode + 1            
        print("The episode is", num_episode)

        new_accumulated_reward = [0,0]
        for i in range(2):
            new_accumulated_reward[i] = accumulated_reward[i] + accumulated_reward[i+2]        
        
        
        with open('pettingzoomagentDMFGAC.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, new_accumulated_reward[0], new_accumulated_reward[1]))
        
        
        team_size = parallel_env.team_size()
       
        print("The number of red melee agents alive is", team_size[0])
        print("The number of red ranged agents alive is", team_size[1])
        print("The number of blue melee agents alive is", team_size[2])
        print("The number of blue ranged agents alive is", team_size[3])

    for agent in parallel_env.agents:
        actor[agent].save_model("./"+agent+"/dmfgacactormodel.ckpt")
        critic[agent].save_model("./"+agent+"/dmfgaccriticmodel.ckpt")
        meanactionobject[agent].save_model("./"+agent+"/dmfgacmeanactionmodel.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v3.parallel_env(map_size = 28, attack_opponent_reward=5)
    parallel_env.seed(1)
    actor = {}
    critic = {}
    meanactionobject = {}
    sess = tf.Session()

    for agent in parallel_env.agents:
        if "mele" in agent:

            name = agent
            actor[agent] = Actor(sess, n_features=5915, n_actions=9, lr=0.00001, name = name)
            critic[agent] = Critic(sess, n_features=5915, n_meanaction=25, lr=0.01, name = name)
            meanactionobject[agent] = Meanaction(sess, n_features=5915, n_meanaction=25, lr=0.01, name = name)
        else: 

            name = agent
            actor[agent] = Actor(sess, n_features=8619, n_actions=25, lr=0.00001, name = name)
            critic[agent] = Critic(sess, n_features=8619, n_meanaction=25, lr=0.01, name = name)
            meanactionobject[agent] = Meanaction(sess, n_features=8619, n_meanaction=25, lr=0.01, name = name)

    sess.run(tf.global_variables_initializer())
    run_battle(parallel_env)


