# Student agent: Add your own agent here
import math
import random
import time

from agents.agent import Agent
from store import register_agent
from utils import MOVES, check_endgame, get_possible_moves_from, get_possible_positions_from

TIME_PER_TURN_GOAL = 1.5 # time per turn that we will try and tune the agent to

@register_agent("minmax_agent")
class MinMaxAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MinMaxAgent, self).__init__()
        self.name = "MinMaxAgent"
        self.autoplay = True
        self.is_first_move = True
        self.C = 100
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
        if self.is_first_move:
            self.tune_c_paramater(TIME_PER_TURN_GOAL, chess_board, my_pos, adv_pos, max_step)
            self.is_first_move = False

        return self.get_best_move(chess_board, my_pos, adv_pos, max_step)
    
    def get_best_move(self, board, my_pos, adv_pos, max_step):
        # expected branching factor based on number of possible moves for each player
        expected_branching_factor = 0.5*(len(get_possible_moves_from(board, my_pos, adv_pos, max_step)) + len(get_possible_moves_from(board, adv_pos, my_pos, max_step)))
        
        # max depth we allow will be proportional to this based on the following equation (check report for derivation), essentially as you near the end of the game, you can look deeper
        max_depth = 2*math.log(self.C, expected_branching_factor)
        
        best_move, _ = min_max(board, my_pos, adv_pos, max_step+1, 0, max_depth=max_depth) # I could not tell you why its max step + 1 but it is
        return best_move

    def tune_c_paramater(self, time_goal, ex_board, ex_my_pos, ex_adv_pos, ex_max_step):
        # tune for 20 seconds or 100 times
        timeout = time.time() + 20
        
        for i in range(100):
            # time one move made
            time_started = time.time()
            self.get_best_move(ex_board, ex_my_pos, ex_adv_pos, ex_max_step)
            time_taken = (time.time() - time_started)

            # tune C
            self.C = self.C*((math.exp((time_goal-time_taken)*math.pow(0.8, i))))
        

            # timeout
            if time.time() > timeout:
                break
        

def rate_move(board, my_pos, adv_pos, max_step, move, turn, max_depth, alpha, beta, current_depth=0):
    # apply move to board
    (x, y), d = move
    board[x, y, d] = True
    board[x + MOVES[d][0], y + MOVES[d][1], (d+2)%4] = True

    if turn==0:
        my_pos = (x,y)
        turn=1
    
    elif turn==1:
        adv_pos = (x,y)
        turn=0
    
    # get board rating
    rating = rate_board(board, my_pos, adv_pos, max_step, turn, max_depth, alpha, beta, current_depth)
    
    # return board to former state
    board[x, y, d] = False
    board[x + MOVES[d][0], y + MOVES[d][1], (d+2)%4] = False
    
    # return rating
    return rating

def rate_board(board, my_pos, adv_pos, max_step, turn, max_depth, alpha, beta, current_depth):
    # check endgame
    is_game_over, my_score, adv_score = check_endgame(board, my_pos, adv_pos)
    
    # if position results in game ending, return score
    if is_game_over:
        # return board to original position
        if my_score > adv_score:
            return 100
        elif my_score < adv_score:
            return -100
        else:
            return 0
    
    # we will recurse with probability max_depth-current_depth:
    recurse_prob = max(0, min(1, max_depth-current_depth))
    
    # if not chosen to recurse, then return naive estimate
    if not random.random() < recurse_prob:
        score = naive_board_estimator(board, my_pos, adv_pos, max_step)
        return score
    
    # otherwise recurse
    _, rating=min_max(board, my_pos, adv_pos, max_step, turn, max_depth=max_depth, alpha=alpha, beta=beta, current_depth=current_depth)

    return rating

def min_max(board, my_pos, adv_pos, max_step, turn, max_depth, current_depth=0, alpha=-999, beta=999):
    # player 1s turn, get max
    if turn==0:
        possible_moves = get_possible_moves_from(board, my_pos, adv_pos, max_step)
        random.shuffle(possible_moves)
        best_move = possible_moves[0]
        for move in possible_moves:
            move_score = rate_move(board, my_pos, adv_pos, max_step, move, turn, max_depth, alpha, beta, current_depth=current_depth+1)
            if move_score > alpha:
                best_move = move
                alpha = move_score
            if alpha >= beta:
                return move, beta
        return best_move, alpha

    # player 2's turn, get minimum
    elif turn==1:
        possible_moves = get_possible_moves_from(board, adv_pos, my_pos, max_step)
        random.shuffle(possible_moves)
        best_move = possible_moves[0]
        for move in possible_moves:
            move_score = rate_move(board, my_pos, adv_pos, max_step, move, turn, max_depth, alpha, beta, current_depth=current_depth+1)
            if move_score < beta:
                best_move = move
                beta = move_score
            if alpha >= beta:
                return move, alpha
        return best_move, beta

def naive_board_estimator(board, my_pos, adv_pos, max_step):
    # naive board value estimation, based on number of possible positions
    return len(get_possible_positions_from(board, my_pos, adv_pos, max_step))-len(get_possible_positions_from(board, adv_pos, my_pos, max_step))