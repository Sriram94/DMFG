from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment

from typing import List, Optional, Dict, Any


class Experience(object):
    """docstring for Experience"""
    envt: Optional[Environment] = None

    def __init__(self, agents: LearningAgent, feasible_actions: List[Action], mean_field: List, time: float, num_requests: int):
        super(Experience, self).__init__()
        self.agents = agents
        self.feasible_actions = feasible_actions
        self.meanfield = mean_field
        self.time = time
        self.num_requests = num_requests

        assert self.envt is not None

        self.representation: Dict[str, Any] = {}
