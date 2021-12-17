from pettingzoo.magent import battle_v2
from RL_braindmfgql import DMFGQL
from RL_actorcritic import Actor
from RL_actorcritic import Critic

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
    n_actions = 21
    num_episode = 0 
    wins = [0,0]
    while num_episode < 100:
        agent_num = 0
        
        observation = parallel_env.reset()
        
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
                if team==0:
                    action = RL_dmfgql[agent].choose_action(agent_observation, estimated_mean_action[agent])
                    actions[agent] = action
                else:
                    action = MFACactor[agent].choose_action(agent_observation)
                    actions[agent] = action
            
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            new_action = actions.copy()



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
                if team == 0:
                    estimated_mean_action[agent] = RL_dmfgql[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])
                    estimated_mean_action[agent] = np.expand_dims(estimated_mean_action[agent], axis = 0)            


            observation = new_observation
            
            
                
            print("The step we are at is", step)
            
            
        

        print("The episode is", num_episode)
        
    
        team_size = parallel_env.team_size()
        print("The number of red agents alive is", team_size[0])
        print("The number of blue agents alive is", team_size[1])
     
        if team_size[0] > team_size[1]: 
            wins[0] = wins[0] + 1
        elif team_size[0] < team_size[1]: 
            wins[1] = wins[1] + 1

        else: 
            if accumulated_reward[0] > accumulated_reward[1]:
                wins[0] = wins[0] + 1
            elif accumulated_reward[0] < accumulated_reward[1]:
                wins[1] = wins[1] + 1
            else: 
                wins[0] = wins[0] + 1
                wins[1] = wins[1] + 1


        num_episode = num_episode + 1            
    
    print('game over')


if __name__ == "__main__":
    parallel_env = battle_v2.parallel_env(map_size = 28, attack_opponent_reward=5)
    parallel_env.seed(1)
    RL_dmfgql = {}
    MFACactor = {}

    
    size = len(parallel_env.agents)
    sess = tf.Session()
    
    for agent in parallel_env.agents: 

        new_name = agent

        if "red" in agent:
            RL_dmfgql[agent] = DMFGQL(21, 6929, sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)
            
            RL_dmfgql[agent].restore_model("./"+agent+"/dmfgqlmodel.ckpt")

            RL_dmfgql[agent].restore_model_meanfield("./"+agent+"/dmfgqlmodelmeanfield.ckpt")
            

        else: 
            MFACactor[agent] = Actor(sess, n_features=6929, n_actions=21, lr=0.00001, name = new_name)

            MFACactor[agent].restore_model("./"+agent+"/actormodel.ckpt")






    sess.run(tf.global_variables_initializer())
    run_battle(parallel_env)


