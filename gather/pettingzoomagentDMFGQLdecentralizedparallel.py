from pettingzoo.magent import gather_v2
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



def run_gather(parallel_env):
    
    step = 0
    with open('pettingzoomagentDMFGQL.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DMFGQL)", "sumofrewards(DMFGQL)"))
    n_actions = 33
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
        
        
        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                action = RL[agent].choose_action(agent_observation, estimated_mean_action[agent])
                actions[agent] = action
            
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            estimated_previous_mean_action = estimated_mean_action.copy()
            observed_previous_mean_action = observed_mean_action.copy() 
            
            new_action = {}

            for agent in parallel_env.agents:
                new_action[agent] = actions[agent]
            
            neighbour_action = parallel_env.get_neighbour_list(new_action, 7, n_actions)
            
            
            for key in neighbour_action:
                observed_mean_action[key] = np.mean(list(map(lambda x: np.eye(n_actions)[x], neighbour_action[key])), axis=0, keepdims=True)            

            
            for agent in parallel_env.agents: 

                accumulated_reward = accumulated_reward + rewards[agent]
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
                
            
        for agent in parallel_env.agents:
            RL[agent].learn()

        print("The episode is", num_episode)
        team_size = parallel_env.team_size()
        accumulated_reward = accumulated_reward/team_size[0]
        with open('pettingzoomagentDMFGQL.csv', 'a') as myfile:
            print("The number of agents alive is", len(parallel_env.agents))
            print("The accumulated reward is", accumulated_reward)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
    
    
    
    for agent in parallel_env.agents:
        RL[agent].save_model("./"+agent+"/dmfgqlmodel.ckpt")
        RL[agent].save_model_meanfield("./"+agent+"/dmfgqlmeanfieldmodel.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = gather_v2.parallel_env()
    parallel_env.seed(1)
    RL = {}
    sess = tf.Session()
    size = len(parallel_env.agents)

    for agent in parallel_env.agents:

        new_name = agent
        RL[agent] = DMFGQL(33,9675,sess, name = new_name)
    
    
    sess.run(tf.global_variables_initializer()) 
    run_gather(parallel_env)


