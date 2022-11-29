from pettingzoo.magent import gather_v2
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



def run_gather(parallel_env):
    
    step = 0
    n_actions = 33
    with open('pettingzoomagentDMFGAC.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DMFGAC)"))
    
    num_episode = 0 
    while num_episode < 2000:
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
            estimated_mean_action[agent] = estimated_mean_action[agent][0]


        estimated_previous_mean_action = estimated_mean_action.copy()
        observed_previous_mean_action = observed_mean_action.copy()






        accumulated_reward = 0
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            action_list = []
            
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                action = actor[agent].choose_action(agent_observation)
                action_list.append(action)
                actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            if not parallel_env.agents:  
                break
            
            new_action = {}
            for agent in parallel_env.agents: 
                new_action[agent] = actions[agent]


            estimated_previous_mean_action = estimated_mean_action.copy()
            observed_previous_mean_action = observed_mean_action.copy()


            neighbour_action = parallel_env.get_neighbour_list(new_action, 7, n_actions)
            
            
            
            
            for key in neighbour_action:
                observed_mean_action[key] = np.mean(list(map(lambda x: np.eye(n_actions)[x], neighbour_action[key])), axis=0, keepdims=True)


            
            for agent in parallel_env.agents: 
                
                accumulated_reward = accumulated_reward + rewards[agent]
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


        print("The episode is", num_episode)
        team_size = parallel_env.team_size()
        accumulated_reward = accumulated_reward/team_size[0]

        with open('pettingzoomagentDMFGAC.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("The amount of food remaining is", parallel_env.food_size())
        print("The number of agents alive is", team_size[0])
    
    
    for agent in actor:
        actor[agent].save_model("./"+agent+"/dmfgacactormodel.ckpt")
        critic[agent].save_model("./"+agent+"/dmfgaccriticmodel.ckpt")
        meanactionobject[agent].save_model("./"+agent+"/dmfgacmeanactionmodel.ckpt")
    
    print('game over')


if __name__ == "__main__":
    parallel_env = gather_v2.parallel_env()
    parallel_env.seed(1)
    sess = tf.Session()
    actor = {} 
    critic = {} 
    meanactionobject = {}

    for agent in parallel_env.agents:
        name = agent
        actor[agent] = Actor(sess, n_features=9675, n_actions=33, lr=0.00001, name = name)
        critic[agent] = Critic(sess, n_features=9675, n_actions=33, lr=0.01, name = name)
        meanactionobject[agent] = Meanaction(sess, n_features=9675, n_actions=33, lr=0.01, name = name)
    
    print("The amount of food remaining is", parallel_env.food_size())
    sess.run(tf.global_variables_initializer())
    run_gather(parallel_env)


