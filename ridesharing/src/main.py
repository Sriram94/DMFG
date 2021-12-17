import csv
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from ValueFunction import PathBasedNN, RewardPlusDelay, NeuralNetworkBased
from Experience import Experience
from Request import Request

from typing import List

import pdb
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import Pool
import argparse


def run_epoch(envt,
              oracle,
              central_agent,
              value_function,
              DAY,
              is_training,
              num_agents,
              agents_predefined=None,
              TRAINING_FREQUENCY: int=1):

    agent_experience = {}
    Experience.envt = envt
    
    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    print("DAY: {}".format(DAY))
    request_generator = envt.get_request_batch(DAY)
    total_value_generated = 0
    num_total_requests = 0
    while True:
        try:
            current_requests = next(request_generator)
            print("Current time: {}".format(envt.current_time))
            print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            break

        scored_actions_all_agents = []
        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)

        for i in range(num_agents):
            agent_experience[i] = Experience(agents[i], feasible_actions_all_agents[i], envt.current_time, len(current_requests))
        for i in range(num_agents):
            scored_actions_all_agents.append(value_function[i].get_value([agent_experience[i]]))


        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)
        
        
        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        if (is_training):
            for i in range(num_agents):
                value_function[i].remember(agent_experience[i])

            for i in range(num_agents):
                if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                    value_function[i].update()

                for action, score in scored_actions_all_agents[0]:
                    print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))
                for idx, (action, score) in enumerate(scored_final_actions[:10]):
                    print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))

        
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Writing statistics to logs
        value_function[0].add_to_logs('rewards_day_{}'.format(envt.num_days_trained), sum(rewards), envt.current_time)
        avg_capacity = sum([agent.path.current_capacity for agent in agents]) / envt.NUM_AGENTS
        value_function[0].add_to_logs('avg_capacity_day_{}'.format(envt.num_days_trained), avg_capacity, envt.current_time)

        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)
        
        ratio = total_value_generated/num_total_requests
        
        with open('ridesharing.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(envt.current_time, ratio))

    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))





    return total_value_generated


if __name__ == '__main__':
    # pdb.set_trace()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', type=int, default=10)
    parser.add_argument('-n', '--numagents', type=int, default=100)
    parser.add_argument('-d', '--pickupdelay', type=int, default=580)
    parser.add_argument('-t', '--decisioninterval', type=int, default=60)
    parser.add_argument('-m', '--modellocation', type=str)
    args = parser.parse_args()

    Request.MAX_PICKUP_DELAY = args.pickupdelay
    Request.MAX_DROPOFF_DELAY = 2 * args.pickupdelay


    with open('ridesharing.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "percentage_request"))


    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    TRAINING_DAYS: List[int] = list(range(3, 10))
    VALID_DAYS: List[int] = [2]
    TEST_DAYS: List[int] = list(range(11, 16))
    VALID_FREQ: int = 4
    SAVE_FREQ: int = VALID_FREQ
    LOG_DIR: str = '../logs/{}agent_{}capacity_{}delay_{}interval/'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval)

    envt = NYEnvironment(args.numagents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=args.decisioninterval)
    oracle = Oracle(envt)
    value_function = {} 
    num_agents = args.numagents
    central_agent = CentralAgent(envt)
    
    for i in range(args.numagents): 
        
        #if args.modellocation:
        #    modellocation = './models/'+str(i)+args.modellocation
        #    value_function[i] = PathBasedNN(envt, log_dir=LOG_DIR, load_model_loc=modellocation)
        #else:
        #    value_function[i] = PathBasedNN(envt, log_dir=LOG_DIR, load_model_loc=args.modellocation)
        
        value_function[i] = RewardPlusDelay(DELAY_COEFFICIENT=1e-7, log_dir=LOG_DIR)
        
    #RewardPlusDelay performs the CO approach in our paper. 
    #PathBasedNN performs the NeurADP approach in our paper. 

    max_test_score = 0
    for epoch_id in range(NUM_EPOCHS):
        for day in TRAINING_DAYS:
            total_requests_served = run_epoch(envt, oracle, central_agent, value_function, day, True, num_agents)
            print("\nDAY: {}, Requests: {}\n\n".format(day, total_requests_served))
            value_function[0].add_to_logs('requests_served', total_requests_served, envt.num_days_trained)

            if (envt.num_days_trained % VALID_FREQ == VALID_FREQ - 1):
                test_score = 0
                for day in VALID_DAYS:
                    total_requests_served = run_epoch(envt, oracle, central_agent, value_function, day, False, num_agents)
                    print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
                    test_score += total_requests_served
                value_function[0].add_to_logs('validation_score', test_score, envt.num_days_trained)

                if (isinstance(value_function[0], NeuralNetworkBased)):
                    if (test_score > max_test_score or (envt.num_days_trained % SAVE_FREQ) == (SAVE_FREQ - 1)):
                        for i in range(num_agents):
                            value_function[i].model.save('./models/'+str(i)+'{}_{}agent_{}capacity_{}delay_{}interval_{}_{}.h5'.format(type(value_function[i]).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, envt.num_days_trained, test_score))
                            print("Saved")
                        max_test_score = test_score if test_score > max_test_score else max_test_score

            envt.num_days_trained += 1


    for day in TEST_DAYS:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, False)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

        total_requests_served = run_epoch(envt, oracle, central_agent, value_function, day, False, num_agents, agents_predefined=agents)
        print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        value_function[0].add_to_logs('test_requests_served', total_requests_served, envt.num_days_trained)


        envt.num_days_trained += 1
