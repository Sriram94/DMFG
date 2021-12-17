from pettingzoo.magent import combined_arms_v3
from RL_braindmfgql import DMFGQL
from RL_dmfgac import Actor
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
    n_actions = 25
    num_episode = 0 
    wins = [0,0]
    while num_episode < 100:
        agent_num = 0
        
        observation = parallel_env.reset()
        list_tmp = [0]
        
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
                    if "mele" in agent:
                        team = 0
                    else:
                        team = 1

                elif "blue" in agent:
                    if "mele" in agent:
                        team = 2
                    else:
                        team = 3


                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if team == 0: 
                    action = RL_dmfgql_melee[agent].choose_action(agent_observation, estimated_mean_action[agent])
                
                elif team == 1: 
                    action = RL_dmfgql_ranged[agent].choose_action(agent_observation, estimated_mean_action[agent])
                
                elif team == 2: 
                    action = actor_dmfgac_melee[agent].choose_action(agent_observation)
                
                elif team == 3: 
                    action = actor_dmfgac_ranged[agent].choose_action(agent_observation)
    
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
                    major = 0
                    if "mele" in agent:
                        team = 0
                    else:
                        team = 2

                elif "blue" in agent:
                    major = 1
                    if "mele" in agent:
                        team = 1
                    else:
                        team = 3
                
                
                
                accumulated_reward[major] = accumulated_reward[major] + rewards[agent]
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)
                if major == 0: 
                    
                    if team == 0: 
                        estimated_mean_action[agent] = RL_dmfgql_melee[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])
                        estimated_mean_action[agent] = np.expand_dims(estimated_mean_action[agent], axis = 0)
                    
                    else: 
                        estimated_mean_action[agent] = RL_dmfgql_ranged[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])
                        estimated_mean_action[agent] = np.expand_dims(estimated_mean_action[agent], axis = 0)




            observation = new_observation
            
            
                
            print("The step we are at is", step)
   
            
            
        team_size = parallel_env.team_size()
        major_team_size = [0,0]

        major_team_size[0] = team_size[0] + team_size[1]
        major_team_size[1] = team_size[2] + team_size[3]

        if major_team_size[0] > major_team_size[1]:
            wins[0] = wins[0] + 1
        elif major_team_size[0] < major_team_size[1]:
            wins[1] = wins[1] + 1

        else:
            if accumulated_reward[0] > accumulated_reward[1]:
                wins[0] = wins[0] + 1
            elif accumulated_reward[0] < accumulated_reward[1]:
                wins[1] = wins[1] + 1
            else:
                wins[0] = wins[0] + 1
                wins[1] = wins[1] + 1



            
        

        print("The episode is", num_episode)
        
        num_episode = num_episode + 1            
    

    
    
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v3.parallel_env(map_size = 28, attack_opponent_reward=5)
    parallel_env.seed(1)

    RL_dmfgql_ranged = {}
    RL_dmfgql_melee = {} 

    actor_dmfgac_ranged = {}
    actor_dmfgac_melee = {} 


    size = len(parallel_env.agents)
    sess = tf.Session()
    
    
    for agent in parallel_env.agents:

        if "red" in agent:
            if "mele" in agent:
                new_name = agent
                RL_dmfgql_melee[agent] = DMFGQL(9,5915,25, sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)

                RL_dmfgql_melee[agent].restore_model("./"+agent+"/dmfgqlmodel.ckpt")
                RL_dmfgql_melee[agent].restore_model_meanfield("./"+agent+"/dmfgqlmeanfieldmodel.ckpt")
            
            else:
                
                new_name = agent
                
                RL_dmfgql_ranged[agent] = DMFGQL(25,8619, 25, sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)
                
                RL_dmfgql_ranged[agent].restore_model("./"+agent+"/dmfgqlmodel.ckpt")
                RL_dmfgql_ranged[agent].restore_model_meanfield("./"+agent+"/dmfgqlmeanfieldmodel.ckpt")


        if "blue" in agent:
            if "mele" in agent:
                new_name = agent
                actor_dmfgac_melee[agent] = Actor(sess, n_features=5915, n_actions=9, lr=0.00001, name = new_name)

                actor_dmfgac_melee[agent].restore_model("./"+agent+"/dmfgacactormodel.ckpt")


            else:
                new_name = agent
                actor_dmfgac_ranged[agent] = Actor(sess, n_features=8619, n_actions=25, lr=0.00001, name = new_name)

                actor_dmfgac_ranged[agent].restore_model("./"+agent+"/dmfgacactormodel.ckpt")

    team_size = parallel_env.team_size()
    print("The number of red melee agents alive is", team_size[0])
    print("The number of red ranged agents alive is", team_size[1])
    print("The number of blue melee agents alive is", team_size[2])
    print("The number of blue ranged agents alive is", team_size[3])

    sess.run(tf.global_variables_initializer())


    run_battle(parallel_env)
