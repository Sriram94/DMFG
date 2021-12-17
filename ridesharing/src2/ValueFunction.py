from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment
from Path import Path
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
from Experience import Experience
from CentralAgent import CentralAgent
from Request import Request
from Experience import Experience

from typing import List, Tuple, Deque, Dict, Any, Iterable

from abc import ABC, abstractmethod
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Masking, Concatenate, Flatten, Bidirectional  
from keras.models import Model, load_model, clone_model  
from keras.backend import function as keras_function  
from keras.optimizers import Adam  
from keras.initializers import Constant  
from tensorflow.summary import FileWriter  
from tensorflow import Summary  
from collections import deque
import numpy as np
from itertools import repeat
from copy import deepcopy
from os.path import isfile, isdir
from os import makedirs
import pickle
import tensorflow as tf

np.random.seed(1)

class ValueFunction(ABC):
    """docstring for ValueFunction"""

    def __init__(self, log_dir: str):
        super(ValueFunction, self).__init__()

        log_dir = log_dir + type(self).__name__ + '/'
        if not isdir(log_dir):
            makedirs(log_dir)
        self.writer = FileWriter(log_dir)

    def add_to_logs(self, tag: str, value: float, step: int) -> None:
        summary = Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    @abstractmethod
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def remember(self, experience: Experience):
        raise NotImplementedError




class NeuralNetworkBased(ValueFunction):
    """docstring for NeuralNetwork"""

    def __init__(self, envt: Environment, load_model_loc: str, load_model_loc2:str, log_dir: str, GAMMA: float=-1, BATCH_SIZE_FIT: int=32, BATCH_SIZE_PREDICT: int=8192, TARGET_UPDATE_TAU: float=0.1):
        super(NeuralNetworkBased, self).__init__(log_dir)

        self.envt = envt
        self.GAMMA = GAMMA if GAMMA != -1 else (1 - (0.1 * 60 / self.envt.EPOCH_LENGTH))
        self.BATCH_SIZE_FIT = BATCH_SIZE_FIT
        self.BATCH_SIZE_PREDICT = BATCH_SIZE_PREDICT
        self.TARGET_UPDATE_TAU = TARGET_UPDATE_TAU

        self._epoch_id = 0

        MIN_LEN_REPLAY_BUFFER = 1e6 / self.envt.NUM_AGENTS
        epochs_in_episode = (self.envt.STOP_EPOCH - self.envt.START_EPOCH) / self.envt.EPOCH_LENGTH
        len_replay_buffer = max((MIN_LEN_REPLAY_BUFFER, epochs_in_episode))
        self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

        self.model: Model = load_model(load_model_loc) if load_model_loc else self._init_NN(self.envt.NUM_LOCATIONS)
        self.model2: Model2 = load_model(load_model_loc) if load_model_loc2 else self._init_NNmeanfield(self.envt.NUM_LOCATIONS)

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model2.compile(optimizer='adam', loss='mean_squared_error')
        
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.update_target_model = self._soft_update_function(self.target_model, self.model)

    def _soft_update_function(self, target_model: Model, source_model: Model) -> keras_function:
        target_weights = target_model.trainable_weights
        source_weights = source_model.trainable_weights

        updates = []
        for target_weight, source_weight in zip(target_weights, source_weights):
            updates.append((target_weight, self.TARGET_UPDATE_TAU * source_weight + (1. - self.TARGET_UPDATE_TAU) * target_weight))

        return keras_function([], [], updates=updates)

    @abstractmethod
    def _init_NN(self, num_locs: int):
        raise NotImplementedError()

    @abstractmethod
    def _init_NNmeanfield(self, num_locs: int):
        raise NotImplementedError()
    
    @abstractmethod
    def _format_input_batch(self, agents: List[List[LearningAgent]], meanfield: List, current_time: float, num_requests: int):
        raise NotImplementedError

    def _get_input_batch_next_state(self, experience: Experience) -> Dict[str, np.ndarray]:
        
        all_agents_post_actions = []
        agent = experience.agents
        feasible_actions = experience.feasible_actions
        agents_post_actions = []
        for action in feasible_actions:
            
            agent_next_time = deepcopy(agent)
            assert action.new_path
            agent_next_time.path = deepcopy(action.new_path)
            self.envt.simulate_motion([agent_next_time], rebalance=False)

            agents_post_actions.append(agent_next_time)
        all_agents_post_actions.append(agents_post_actions)
        next_time = experience.time + self.envt.EPOCH_LENGTH

        return self._format_input_batch(all_agents_post_actions, experience.meanfield, next_time, experience.num_requests)

    def _flatten_NN_input(self, NN_input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        shape_info: List[int] = []

        for key, value in NN_input.items():
            
            if not shape_info:
                cumulative_sum = 0
                shape_info.append(cumulative_sum)
                for idx, list_el in enumerate(value):
                    cumulative_sum += len(list_el)
                    shape_info.append(cumulative_sum)

            NN_input[key] = np.array([element for array in value for element in array])

        return NN_input, shape_info

    def _reconstruct_NN_output(self, NN_output: np.ndarray, shape_info: List[int]) -> List[List[int]]:
        
        NN_output = NN_output.flatten()

        
        assert shape_info
        output_as_list = []
        for idx in range(len(shape_info) - 1):
            start_idx = shape_info[idx]
            end_idx = shape_info[idx + 1]
            list_el = NN_output[start_idx:end_idx].tolist()
            output_as_list.append(list_el)

        return output_as_list

    def _format_experiences(self, experiences: List[Experience], is_current: bool) -> Tuple[Dict[str, np.ndarray], List[int]]:
        action_inputs = None
        for experience in experiences:
            agent = experience.agents
            if not (self.__class__.__name__ in experience.representation):
                experience.representation[self.__class__.__name__] = self._get_input_batch_next_state(experience)

            if is_current:
                batch_input = self._format_input_batch([[agent]], experience.meanfield, experience.time, experience.num_requests)
            else:
                batch_input = deepcopy(experience.representation[self.__class__.__name__])

            if action_inputs is None:
                action_inputs = batch_input
            else:
                for key, value in batch_input.items():
                    action_inputs[key].extend(value)

        assert action_inputs is not None
        return self._flatten_NN_input(action_inputs)

    def get_value(self, experiences: List[Experience], network: Model=None) -> List[Tuple[Action, float]]:
        
        action_inputs, shape_info = self._format_experiences(experiences, is_current=False)
        
        if (network is None):
            expected_future_values = self.model.predict(action_inputs, batch_size=self.BATCH_SIZE_PREDICT)
        else:
            expected_future_values = network.predict(action_inputs, batch_size=self.BATCH_SIZE_PREDICT)

        expected_future_values = self._reconstruct_NN_output(expected_future_values, shape_info)
        
        
        def get_score(action: Action, value: float):
            return self.envt.get_reward(action) + self.GAMMA * value

        feasible_actions = [feasible_actions for experience in experiences for feasible_actions in experience.feasible_actions]
        
        scored_actions: List[Tuple[Action, float]] = []
        
        i = 0 
        for action in feasible_actions:
            value = expected_future_values[0][i]
            tmp_scored_actions = (action, get_score(action, value)) 
            scored_actions.append(tmp_scored_actions)
            i = i+1
        
        return scored_actions




    def get_meanfield(self, experiences: List[Experience], network: Model=None) -> List[Tuple[Action, float]]:
        action_inputs, shape_info = self._format_experiences(experiences, is_current=False)
        if (network is None):
            new_meanfield = self.model2.predict(action_inputs)
        else:
            new_meanfield = network.predict(action_inputs)
        
        new_meanfield = new_meanfield.tolist()
        
        return new_meanfield
        





    def remember(self, experience: Experience):
        self.replay_buffer.add(experience)

    
    def update(self, num_samples: int = 3):
        
        num_min_train_samples = int(5e2)
        if (num_min_train_samples > len(self.replay_buffer)):
            return

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
            experiences, weights, batch_idxes = self.replay_buffer.sample(num_samples, beta)
        else:
            experiences = self.replay_buffer.sample(num_samples)
            weights = None

        for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
            if weights is not None:
                weights_new = np.array([weights[experience_idx]])
            scored_actions = self.get_value([experience], network=self.target_model)  
            

            

            new_score = 0
            for agent_idx, (action, score) in enumerate(scored_actions):
                new_score = new_score + score
            new_score = new_score/len(scored_actions)
            new_list = []
            new_list.append(new_score)



            action_inputs, _ = self._format_experiences([experience], is_current=True)
            action_inputs["meanfield_input"] = action_inputs["meanfield_input"][0]
            action_inputs["meanfield_input"] = np.expand_dims(action_inputs["meanfield_input"], axis=0)
            
            supervised_targets = np.array(new_list).reshape((-1, 1))
            
            history = self.model.fit(action_inputs, supervised_targets, batch_size=self.BATCH_SIZE_FIT, sample_weight=weights_new)
            loss = history.history['loss'][-1]
            self.add_to_logs('loss', loss, self._epoch_id)

            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                predicted_values = self.model.predict(action_inputs, batch_size=self.BATCH_SIZE_PREDICT)
                loss = np.mean((predicted_values - supervised_targets) ** 2 + 1e-6)
                self.replay_buffer.update_priorities([batch_idx], [loss])

            self.update_target_model([])

            self._epoch_id += 1


    def updatemeanfield(self, experiences, target):
        new_target = [] 
        for i in range(len(experiences[0].feasible_actions)):
            new_target.append(target)
        targetarray = np.array(new_target)
        action_inputs, _ = self._format_experiences(experiences, is_current=False)
        self.model2.fit(action_inputs, targetarray)







class DMFGQL(NeuralNetworkBased):

    def __init__(self, envt: Environment, load_model_loc: str='', load_model_loc2: str='', log_dir: str='../logs/'):
        super(DMFGQL, self).__init__(envt, load_model_loc, load_model_loc2, log_dir)

    def _init_NN(self, num_locs: int) -> Model:
        if (isfile(self.envt.DATA_DIR + 'embedding_weights.pkl')):
            weights = pickle.load(open(self.envt.DATA_DIR + 'embedding_weights.pkl', 'rb'))
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding', embeddings_initializer=Constant(weights[0]), trainable=False)
        else:
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding')

        path_location_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32', name='path_location_input')
        path_location_embed = location_embed(path_location_input)

        delay_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1), name='delay_input')
        delay_masked = Masking(mask_value=-1)(delay_input)

        path_input = Concatenate()([path_location_embed, delay_masked])
        path_embed = LSTM(200, go_backwards=True)(path_input)

        current_time_input = Input(shape=(1,), name='current_time_input')
        current_time_embed = Dense(100, activation='elu', name='time_embedding')(current_time_input)

        meanfield_input = Input(shape=(4461,), name='meanfield_input')
        meanfield_embed = Dense(100, activation='elu', name='meanfield_embedding')(meanfield_input)

        num_requests_input = Input(shape=(1,), name='num_requests_input')

        state_embed = Concatenate()([path_embed, current_time_embed, meanfield_embed, num_requests_input])
        state_embed = Dense(300, activation='elu', name='state_embed_1')(state_embed)
        state_embed = Dense(300, activation='elu', name='state_embed_2')(state_embed)

        output = Dense(1, name='output')(state_embed)

        model = Model(inputs=[path_location_input, delay_input, current_time_input, meanfield_input, num_requests_input], outputs=output)

        return model

    def _init_NNmeanfield(self, num_locs: int) -> Model:
        if (isfile(self.envt.DATA_DIR + 'embedding_weights_meanfield.pkl')):
            weights = pickle.load(open(self.envt.DATA_DIR + 'embedding_weights_meanfield.pkl', 'rb'))
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding', embeddings_initializer=Constant(weights[0]), trainable=False)
        else:
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding')

        path_location_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32', name='path_location_input')
        path_location_embed = location_embed(path_location_input)

        delay_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1), name='delay_input')
        delay_masked = Masking(mask_value=-1)(delay_input)

        path_input = Concatenate()([path_location_embed, delay_masked])
        path_embed = LSTM(200, go_backwards=True)(path_input)

        current_time_input = Input(shape=(1,), name='current_time_input')
        current_time_embed = Dense(100, activation='elu', name='time_embedding')(current_time_input)

        meanfield_input = Input(shape=(4461,), name='meanfield_input')
        meanfield_embed = Dense(100, activation='elu', name='meanfield_embedding')(meanfield_input)

        num_requests_input = Input(shape=(1,), name='num_requests_input')

        state_embed = Concatenate()([path_embed, current_time_embed, meanfield_embed, num_requests_input])
        state_embed = Dense(50, activation='relu', name='state_embed_1')(state_embed)
        state_embed = Dense(50, activation='relu', name='state_embed_2')(state_embed)
        state_embed = Dense(50, activation='relu', name='state_embed_3')(state_embed)

        output = Dense(4461, activation='softmax', name='output')(state_embed)

        model = Model(inputs=[path_location_input, delay_input, current_time_input, meanfield_input, num_requests_input], outputs=output)

        return model
    def _format_input(self, agent: LearningAgent, meanfield: List, current_time: float, num_requests: float, num_other_agents: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        current_time_input = (current_time - self.envt.START_EPOCH) / (self.envt.STOP_EPOCH - self.envt.START_EPOCH)
        num_requests_input = num_requests / self.envt.NUM_AGENTS
        num_other_agents_input = num_other_agents / self.envt.NUM_AGENTS
        
        location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32')
        delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1)) - 1

        location_order[0] = agent.position.next_location + 1
        delay_order[0] = 1

        for idx, node in enumerate(agent.path.request_order):
            if (idx >= 2 * self.envt.MAX_CAPACITY):
                break

            location, deadline = agent.path.get_info(node)
            visit_time = node.expected_visit_time

            location_order[idx + 1] = location + 1
            delay_order[idx + 1, 0] = (deadline - visit_time) / Request.MAX_DROPOFF_DELAY  

        return location_order, delay_order, meanfield, current_time_input, num_requests_input, num_other_agents_input

    def _format_input_batch(self, all_agents_post_actions: List[List[LearningAgent]], meanfield: List, current_time: float, num_requests: int) -> Dict[str, Any]:
        input: Dict[str, List[Any]] = {"path_location_input": [], "delay_input": [], "meanfield_input": [], "current_time_input": [], "num_requests_input": []}

        for agent_post_actions in all_agents_post_actions:
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_agents_input = []
            meanfield_input = []

            current_agent = agent_post_actions[0]  
            num_other_agents = 0
            for other_agents_post_actions in all_agents_post_actions:
                other_agent = other_agents_post_actions[0]
                if (self.envt.get_travel_time(current_agent.position.next_location, other_agent.position.next_location) < Request.MAX_PICKUP_DELAY or
                        self.envt.get_travel_time(other_agent.position.next_location, current_agent.position.next_location) < Request.MAX_PICKUP_DELAY):
                    num_other_agents += 1

            for agent in agent_post_actions:
                location_order, delay_order, meanfield, current_time_scaled, num_requests_scaled, num_other_agents_scaled = self._format_input(agent, meanfield, current_time, num_requests, num_other_agents)

                current_time_input.append(current_time_scaled)
                num_requests_input.append(num_requests)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_agents_input.append(num_other_agents_scaled)
                if len(meanfield) == 4461:
                    meanfield_input.append(meanfield)
            
                else:
                    meanfield_input = meanfield.copy()

            
            input["current_time_input"].append(current_time_input)
            input["num_requests_input"].append(num_requests_input)
            input["delay_input"].append(delay_input)
            input["path_location_input"].append(path_location_input)
            input["meanfield_input"].append(meanfield_input)

        return input

