from pettingzoo.magent import battle_v2
from RL_braindmfgql import DMFGQL
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
    
    step = 0
    with open('pettingzoomagentDMFGQL.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DMFGQL)", "sumofrewards(DMFGQL)"))
    n_actions = 21
    num_episode = 0 
    while num_episode < 2000:
        agent_num = 0
        
        observation = parallel_env.reset()
        list_tmp = []
        for i in range(n_actions):
            list_tmp.append(i)
        
        estimated_mean_action = {}
        estimated_previous_mean_action = {}
        observed_previous_mean_action = {} 
        observed_mean_action = {}
        
        for agent in parallel_env.agents:
            estimated_mean_action[agent] = np.mean(list(map(lambda x: np.eye(n_actions)[x], list_tmp)), axis=0, keepdims=True)
            observed_mean_action[agent] = np.mean(list(map(lambda x: np.eye(n_actions)[x], list_tmp)), axis=0, keepdims=True)
            

        estimated_previous_mean_action = estimated_mean_action.copy()
        observed_previous_mean_action = observed_mean_action.copy()


        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                if "red" in agent:
                    team = 0
                else: 
                    team = 1
                
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                action = RL[agent].choose_action(agent_observation, estimated_mean_action[agent])
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
                    team = 0
                else: 
                    team = 1
                
                accumulated_reward[team] = accumulated_reward[team] + rewards[agent]
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)
                RL[agent].learn_mean_action(agent_observation, observed_mean_action[agent], observed_previous_mean_action[agent]) 
                estimated_mean_action[agent] = RL[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])
                estimated_mean_action[agent] = np.expand_dims(estimated_mean_action[agent], axis = 0)            
                RL[agent].store_transition(agent_observation, actions[agent], estimated_previous_mean_action[agent], estimated_mean_action[agent], rewards[agent], agent_nextobservation)



            observation = new_observation
            
            
                
            print("The step we are at is", step)
            
            

        for agent in RL:
            RL[agent].learn() 

        print("The episode is", num_episode)
        
        with open('pettingzoomagentDMFGQL.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
    
        team_size = parallel_env.team_size()
        print("The number of red agents alive is", team_size[0])
        print("The number of blue agents alive is", team_size[1])
        total_agents = team_size[0] + team_size[1]
        
    
        num_episode = num_episode + 1            


    for agent in RL:
        RL[agent].save_model("./"+agent+"/dmfgqlmodel.ckpt")
        RL[agent].save_model_meanfield("./"+agent+"/dmfgqlmodelmeanfield.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = battle_v2.parallel_env(map_size = 28)
    
    parallel_env.seed(1)
    RL = {}
    sess = tf.Session()
    size = len(parallel_env.agents)


    for agent in parallel_env.agents:

        new_name = agent
        RL[agent] = DMFGQL(21,6929,sess, name = new_name)



    sess.run(tf.global_variables_initializer())
    run_battle(parallel_env)


