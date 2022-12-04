# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
from functools import partial
import sys
import random
import math


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
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
        possible_moves = self.get_possible_moves_from(chess_board, my_pos, adv_pos, max_step)
        move_rater = partial(self.move_rating, chess_board, adv_pos, max_step)
        best_move = max(possible_moves, key=move_rater)
        return best_move

    def move_rating(self, chess_board, adv_pos, max_step, move):
        board_after_move = deepcopy(chess_board)
        (x, y), d = move
        board_after_move[x, y, d] = True
        board_after_move[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = True
        return self.board_rating(board_after_move, (x,y), adv_pos, max_step)

    def board_rating(self, board, my_pos, adv_pos, max_step):
        # rating will be the number of moves we have availible - number of moves openent has availible
        my_possible_moves = len(self.get_possible_moves_from(board, my_pos, adv_pos, max_step))
        adv_possible_moves = len(self.get_possible_moves_from(board, adv_pos, my_pos, max_step))
        
        return my_possible_moves - adv_possible_moves

    def get_possible_moves_from(self, chess_board, pos, adv_pos, steps, visited=None):
        if visited is None:
            visited = {}
        # check if position is out of board, or oponent's square
        if not self.in_board(chess_board, pos) or pos == adv_pos:
            return []

        # check if already visited position with equal or more steps
        if visited.get(pos, 0) >= steps:
            return []

        # add self to visited
        visited[pos] = steps
        
        # add possible moves
        possible_moves = []
        for i in range(4):
            if not chess_board[pos[0], pos[1], i]:
                possible_moves.append((pos, i))
        
        # base case if steps is 0 return self only
        if steps == 0:
            return possible_moves
    
        # recursively visit all posiible adj moves with one less step
        for i, move in enumerate(self.moves):
            if not chess_board[pos[0], pos[1], i]:
                possible_moves = possible_moves + self.get_possible_moves_from(chess_board, (pos[0] + move[0], pos[1] + move[1]), adv_pos, steps-1, visited)
        
        # remove duplicates
        return list(set(possible_moves))

    def in_board(self, chess_board, pos):
        return 0 <= pos[0] < chess_board.shape[0] and 0 <= pos[1] < chess_board.shape[1]
