# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
from functools import partial
import sys
import random
import math

MAX_DEPTH = 4


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
        self.max_depth=MAX_DEPTH
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
        # board = chess_board, my_pos=my_pos, adv_pos=adv_pos, max_step-max_step, turn=0, current_depth=0, alpha=-999, beta=999
        move_rater = partial(self.move_rating, chess_board, my_pos, adv_pos, max_step, 0, 0, -999, 999)
        best_move = max(possible_moves, key=lambda x : move_rater(x)[0])
        return best_move

    def move_rating(self, board, my_pos, adv_pos, max_step, turn, current_depth, alpha, beta, move):
        # apply move to board
        (x, y), d = move
        board[x, y, d] = True
        board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = True
        # get rating for move
        if turn==0:
            rating, alpha, beta=self.board_rating(board, (x,y), adv_pos, max_step, 1, current_depth+1, alpha, beta)
        else:
            rating, alpha, beta=self.board_rating(board, my_pos, (x,y), max_step, 0, current_depth+1, alpha, beta)
        # undo moves
        board[x, y, d] = False
        board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = False
        # return rating
        return rating, alpha, beta

    def board_rating(self, board, my_pos, adv_pos, max_step, turn, current_depth):
        # rating will be the number of moves we have availible - number of moves openent has availible
        is_game_over, my_score, adv_score = self.check_endgame(board, my_pos, adv_pos)
        
        # if position results in game ending, return winner
        if is_game_over:
            if my_score > adv_score:
                return 999
            else:
                return -999
        
        # if max depth use naive estimator
        if current_depth > self.max_depth:
            return self.naive_board_estimator(board, my_pos, adv_pos, max_step)

        if turn==0:
            possible_moves = self.get_possible_moves_from(board, my_pos, adv_pos, max_step)
        else:
            possible_moves = self.get_possible_moves_from(board, adv_pos, my_pos, max_step)

        # shuffle possible moves
        possible_moves = random.shuffle(possible_moves)

        for move in possible_moves:
            if turn==0:
                best_score = -999
            else:
                best_score = 999
            
            move_score, alpha, beta = self.move_rating(board=board, my_pos=my_pos, adv_pos=adv_pos, max_step=max_step, turn=turn, current_depth=current_depth, alpha=alpha, beta=beta, move=move)
            # alpha-beta pruning
            if turn==0:
                if move_score > best_score:
                    best_score = move_score
                # if move_score < alpha:
                #     break
            else:
                if move_score < best_score:
                    best_score = move_score
                # if move_score > beta:
                #     break
        
        if turn==0:
            beta = min(beta, best_score)
        else:
            alpha = max(alpha, best_score)
        
        return best_score, alpha, beta

    def naive_board_estimator(self, board, my_pos, adv_pos, max_step):
        # naive board value estimation, based on number of possible moves
        my_possible_moves = len(self.get_possible_moves_from(board, my_pos, adv_pos, max_step))
        adv_possible_moves = len(self.get_possible_moves_from(board, adv_pos, my_pos, max_step))
        
        return my_possible_moves-adv_possible_moves

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

    def check_endgame(self, board, my_pos, adv_pos):
        # check board to identify if the game is ended (use code from other file)
        board_size = board.shape[0]

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        my_r = find(tuple(my_pos))
        adv_r = find(tuple(adv_pos))
        my_score = list(father.values()).count(my_r)
        adv_score = list(father.values()).count(adv_r)
        if my_r == adv_r:
            return False, my_score, adv_score
        return True, my_score, adv_score

    def in_board(self, chess_board, pos):
        return 0 <= pos[0] < chess_board.shape[0] and 0 <= pos[1] < chess_board.shape[1]
