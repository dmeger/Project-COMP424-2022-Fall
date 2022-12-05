# Student agent: Add your own agent here
import random

from agents.agent import Agent
from store import register_agent
from utils import get_possible_moves_from

@register_agent("my_random_agent")
class MyRandomAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MyRandomAgent, self).__init__()
        self.name = "MyRandomAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        return random.choice(get_possible_moves_from(chess_board, my_pos, adv_pos, max_step+1))