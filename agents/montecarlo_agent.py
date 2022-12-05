# Student agent: Add your own agent here
from copy import deepcopy

import random

from agents.agent import Agent
from store import register_agent
from utils import MOVES, OPPOSITES, get_possible_moves_from, check_endgame

RANDOM_TRIALS = 5
@register_agent("montecarlo_agent")
class MonteCarloAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MonteCarloAgent, self).__init__()
        self.name = "MonteCarloAgent"
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
        max_step = max_step +1
        possible_moves = get_possible_moves_from(chess_board, my_pos, adv_pos, max_step)
        move_scores = [score_move(chess_board, my_pos, adv_pos, max_step, move) for move in possible_moves]
        best_move = max(zip(move_scores, possible_moves))
        print(list(zip(possible_moves, move_scores)))
        print(max(move_scores))
        return best_move[1]

def score_move(board, my_pos, adv_pos, max_steps, move):
    # apply move to board
    (x, y), d = move
    boardcopy = deepcopy(board)
    boardcopy[x, y, d] = True
    boardcopy[x + MOVES[d][0], y + MOVES[d][1], (d+2)%4] = True
    rating=0
    for i in range(RANDOM_TRIALS):
        rating += random_expansion(boardcopy, my_pos, adv_pos, max_steps, turn=1)
    return rating

def random_expansion(board, my_pos, adv_pos, max_steps, turn):
    boardcopy = deepcopy(board)
    game_is_over = False
    
    while not game_is_over:
        if turn == 0:
            (x, y), d = random_walk(board, my_pos, adv_pos, max_steps)
            boardcopy[x, y, d] = True
            boardcopy[x + MOVES[d][0], y + MOVES[d][1], (d+2)%4] = True
            my_pos = (x,y)
            turn=1
        elif turn == 1:
            (x, y), d = random_walk(board, adv_pos, my_pos, max_steps)
            boardcopy[x, y, d] = True
            boardcopy[x + MOVES[d][0], y + MOVES[d][1], (d+2)%4] = True
            adv_pos = (x,y)
            turn=0
        game_is_over, my_score, adv_score = check_endgame(boardcopy, my_pos, adv_pos)

    if my_score == adv_score:
        return 0
    if my_score > adv_score:
        return 1
    return -1

def random_walk(board, my_pos, adv_pos, max_steps):
    # method to generate a random move faster
    opposite_dir = -1
    steps = random.randint(0,max_steps)
    
    for _ in range(steps):
        # choose random valid direction
        dir_choices = [i for i in range(4) if not board[my_pos[0], my_pos[1], i]]
        
        # choose random valid move excluding moving back to previous spot
        move_dir_choices = [dir for dir in dir_choices if (not dir==opposite_dir and not (my_pos[0] + MOVES[dir][0], my_pos[1] + MOVES[dir][1])==adv_pos)]
        
        # if no moves break
        if not move_dir_choices:
            break

        dir_to_move = random.choice(move_dir_choices)
        opposite_dir = OPPOSITES[dir_to_move]

        my_pos = (my_pos[0] + MOVES[dir_to_move][0], my_pos[1] + MOVES[dir_to_move][1])
    
    dir_choices = [i for i in range(4) if not board[my_pos[0], my_pos[1], i]]
    return my_pos, random.choice(dir_choices)
