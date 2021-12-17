from pettingzoo.sisl.waterworld import waterworld
from RL_dmfgac import Actor
from RL_dmfgac import Meanaction
from RL_dmfgac import Critic
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




def run_waterworld(parallel_env):
    
    n_actions = 25
    with open('pettingzoomagentDMFGAC.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DMFGAC)"))
    
    num_episode = 0 
    while num_episode < 2000:
        observation = parallel_env.reset()
        estimated_mean_action = {}
        estimated_previous_mean_action = {}
        previous_observed_mean_action = {}
        observed_mean_action = {}

        mean_action = []
        for i in range(len(parallel_env.agents)*2):
            mean_action.append(0)


        for agent in parallel_env.agents:
            estimated_mean_action[agent] = mean_action 
            observed_mean_action[agent] = mean_action


        estimated_previous_mean_action = estimated_mean_action.copy()
        observed_previous_mean_action = observed_mean_action.copy()



        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        number_caught = 0
        previous_mean = {}
        for step in range(max_cycles):
            action_list = []
            
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                action = actor[agent].choose_action(agent_observation)
                for s in range(len(action)):
                    action_list.append(action[s])
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            number_caught = number_caught + parallel_env.return_number_caught()

            if not parallel_env.agents:  
                break
            


            estimated_previous_mean_action = estimated_mean_action.copy()
            observed_previous_mean_action = observed_mean_action.copy()

            for agent in parallel_env.agents:
                observed_mean_action[agent] = action_list 


            
            for agent in parallel_env.agents: 
                
                accumulated_reward = accumulated_reward + rewards[agent]
                agent_observation = observation[agent]
                agent_nextobservation = new_observation[agent]
                meanactionobject[agent].learn_mean_action(agent_observation, observed_mean_action[agent], observed_previous_mean_action[agent])
                estimated_mean_action[agent] = meanactionobject[agent].get_mean_action(agent_nextobservation, observed_mean_action[agent])

                td_error = critic[agent].learn(agent_observation, estimated_previous_mean_action[agent], estimated_mean_action[agent], rewards[agent], agent_nextobservation)
                actor[agent].learn(agent_observation, actions[agent], td_error)
            
            
            observation = new_observation
            
            
                
            print("The step we are at is", step)


        #print("the total food captured is", number_caught)
        num_episode = num_episode + 1            
        print("The episode is", num_episode)

        with open('pettingzoomagentDMFGAC.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    
    
    for agent in parallel_env.agents:
        actor[agent].save_model("./"+agent+"/dmfgacactormodel.ckpt")
        critic[agent].save_model("./"+agent+"/dmfgaccriticmodel.ckpt")
        meanactionobject[agent].save_model("./"+agent+"/dmfgacmeanactionmodel.ckpt")
 
    print('game over')


if __name__ == "__main__":
    parallel_env = waterworld.parallel_env(n_pursuers=25, n_evaders=25, encounter_reward=1)
    parallel_env.reset()

    actor = {}
    critic = {}
    meanactionobject = {}
    sess = tf.Session()
    size = len(parallel_env.agents)
    action_bound = [-1, 1]
    for agent in parallel_env.agents:
        name = agent

        actor[agent] = Actor(sess, n_features=212, action_bound = action_bound, lr=0.00001, name = name)
        critic[agent] = Critic(sess, n_features=212, n_meanaction=50, lr=0.01, name = name)
        meanactionobject[agent] = Meanaction(sess, n_features=212, n_meanaction=50, lr=0.01, name = name)
    
    sess.run(tf.global_variables_initializer())
    run_waterworld(parallel_env)


