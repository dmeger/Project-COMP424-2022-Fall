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
        self.is_first_move = True
        self.C = 10
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        return random.choice(get_possible_moves_from(chess_board, my_pos, adv_pos, max_step+1))