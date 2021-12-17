from pettingzoo.magent import gather_v2
from RL_braindmfgql import DMFGQL 
from RL_brainDQN import DeepQNetwork
from RL_brainMFQ import MFQ
from RL_dmfgac import Actor
from RL_actorcritic import Actor as MFACActor
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
    n_actions = 33
    num_episode = 0 
    wins = [0,0,0,0,0]
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
        
        
        accumulated_reward = [0,0,0,0,0]
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                
                if classification_dict[agent] == 1: 
                    action = RL_dmfgql[agent].choose_action(agent_observation, estimated_mean_action[agent])
                elif classification_dict[agent] == 2:
                    action = RL_mfq[agent].choose_action(agent_observation, observed_mean_action[agent])
                elif classification_dict[agent] == 3:
                    action = RL_dqn[agent].choose_action(agent_observation)
                elif classification_dict[agent] == 4:
                    action = mfacactor[agent].choose_action(agent_observation)
                elif classification_dict[agent] == 5:
                    action = actor[agent].choose_action(agent_observation)


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

                if classification_dict[agent] == 1:
                    accumulated_reward[0] = accumulated_reward[0] + rewards[agent]
                
                elif classification_dict[agent] == 2:
                    accumulated_reward[1] = accumulated_reward[1] + rewards[agent]
                
                
                elif classification_dict[agent] == 3:
                    accumulated_reward[2] = accumulated_reward[2] + rewards[agent]
                
                elif classification_dict[agent] == 4:
                    accumulated_reward[3] = accumulated_reward[3] + rewards[agent]
                
                elif classification_dict[agent] == 5:
                    accumulated_reward[4] = accumulated_reward[4] + rewards[agent]



                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)
                
                
                
                if classification_dict[agent] == 1:
                    estimated_mean_action[agent] = RL_dmfgql[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])
                    estimated_mean_action[agent] = np.expand_dims(estimated_mean_action[agent], axis = 0)




            observation = new_observation
            print("The step we are at is", step)
            


        num_episode = num_episode + 1            
        
        print("The episode is", num_episode)
        max_value = max(accumulated_reward)
        max_index = accumulated_reward.index(max_value)
        wins[max_index] = wins[max_index] + 1
            
    print('game over')


if __name__ == "__main__":
    parallel_env = gather_v2.parallel_env()
    parallel_env.seed(1)
    RL_dmfgql = {} 
    RL_dqn = {} 
    RL_mfq = {} 
    actor = {} 

    mfacactor = {}
    classification_dict = {}
    sess = tf.Session()
    count = 0
    for agent in parallel_env.agents:
        new_name = agent 
        if count < 6: 
            classification_dict[agent] = 1
            RL_dmfgql[agent] = DMFGQL(33,9675,sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)


            RL_dmfgql[agent].restore_model("./"+agent+"/dmfgqlmodel.ckpt")
            RL_dmfgql[agent].restore_model_meanfield("./"+agent+"/dmfgqlmeanfieldmodel.ckpt")
            
            count = count + 1

        elif count < 12: 
            classification_dict[agent] = 2
            RL_mfq[agent] = MFQ(33, 9675, sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)


            RL_mfq[agent].restore_model("./"+agent+"/mfqmodel.ckpt")

            count = count + 1
         
        elif count < 18: 
            classification_dict[agent] = 3
        
            RL_dqn[agent] = DeepQNetwork(33,9675,sess, e_greedy=1, name = new_name, e_greedy_increment=None, execution=True)


            RL_dqn[agent].restore_model("./"+agent+"/dqnmodel.ckpt")


            count = count + 1
        
        
        elif count < 24: 
            classification_dict[agent] = 4
            
            mfacactor[agent] = MFACActor(sess, n_features=9675, n_actions=33, lr=0.00001, name = new_name)
    
            mfacactor[agent].restore_model("./"+agent+"/actormodel.ckpt")

            count = count + 1
        
        
        
        elif count < 30: 
            classification_dict[agent] = 5
            
            actor[agent] = Actor(sess, n_features=9675, n_actions=33, lr=0.00001, name = new_name)
            actor[agent].restore_model("./"+agent+"/dmfgacactormodel.ckpt")
            
            count = count + 1
   

    sess.run(tf.global_variables_initializer()) 
    run_gather(parallel_env)


