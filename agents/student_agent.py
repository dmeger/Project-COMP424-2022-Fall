# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import copy
import math
import time

class A_Star_Node():

  def __init__(self, pos, parent=None):

    self.pos    = pos
    self.parent = parent
    self.g      = 0
    self.h      = 0

def deepcopy_np_array(arr):
    deepcopy = copy.deepcopy(arr.tolist())
    return np.array(deepcopy)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def distance_to_position(self, my_pos, adv_pos, chess_board, max_dfs_depth=8):
        
        board_size = len(chess_board)

        distances = np.zeros((board_size, board_size))
        
        queue = [(my_pos, 0)]
        encountered = set()
        encountered.add(my_pos)

        while len(queue) > 0:

            cur_position, distance = queue.pop()
            distances[cur_position] = distance

            if distance >= max_dfs_depth:
                continue

            for next_position in self.valid_next_position((cur_position, adv_pos, chess_board), False):
                
                if next_position not in encountered:
                    queue.append([next_position, distance + 1])
                    encountered.add(next_position)
        
        return distances
    
    # Return the heuristic value of a given move
    def utility_of_state(self, my_pos, adv_pos, chess_board):

        dist     = self.distance_to_position(my_pos, adv_pos, chess_board)
        opp_dist = self.distance_to_position(adv_pos, my_pos, chess_board)

        diff = dist - opp_dist

        return np.sum(diff > 0)

    def a_star_heuristic(self, my_pos, adv_pos):
        return abs(my_pos[0]-adv_pos[0]) + abs(my_pos[1]-adv_pos[1])

    def a_star(self, chess_board, my_pos, adv_pos, used_squares, max_step):

        start = A_Star_Node(my_pos)
        start.g = 0
        start.h = self.a_star_heuristic(my_pos, adv_pos)
        end     = A_Star_Node(adv_pos)

        queue = set()
        explored = set()
        queue.add((start, start.g + start.h))

        while len(queue) > 0:

            # Retrieve the lowest f = g + h value node from the queue
            current_node = min(queue, key = lambda t:t[1])

            # Remove the node from the queue
            queue.remove(current_node)

            current_a_star_node = current_node[0]

            if current_a_star_node.pos in explored:
                continue

            # Mark the node as explored
            explored.add(current_a_star_node.pos)

            # Reached the goal
            if current_a_star_node.pos == end.pos:
                path = [current_a_star_node.pos]

                unique_square = False

                # Backtrack through the path till the start point has been reached
                while current_a_star_node.parent != None:

                    if not current_a_star_node in used_squares:
                        unique_square = True

                    path.append(current_a_star_node.parent.pos)
                    current_a_star_node = current_a_star_node.parent
                    
                # Add the configured path to the list of paths
                if not unique_square:
                    return None

                return path[::-1]
            
            # Retrieve the neighbours of the current node
            next_positions = self.valid_next_position((current_a_star_node.pos, adv_pos, chess_board), True)

            queued = False
            for child in next_positions:
                
                if child not in explored and (child not in used_squares or current_a_star_node.g + 1 > max_step):

                    queued = True
                    child_node = A_Star_Node(child, current_a_star_node)
                    child_node.g = current_a_star_node.g + 1
                    child_node.h = self.a_star_heuristic(child_node.pos, adv_pos)
                    queue.add((child_node, child_node.g + child_node.h))

            if not queued:
                
                for child in next_positions:
                
                    if child not in explored:
                        child_node = A_Star_Node(child, current_a_star_node)
                        child_node.g = current_a_star_node.g + 1
                        child_node.h = self.a_star_heuristic(child_node.pos, adv_pos)
                        queue.add((child_node, child_node.g + child_node.h))
        
        return None

    def k_shortest_paths(self, chess_board, my_pos, adv_pos, k, max_step):

        paths = []
        used_squares = set()

        while len(paths) < k:

            path = self.a_star(chess_board, my_pos, adv_pos, used_squares, max_step)
            
            if path == None:
                break

            for i in range(1, len(path) - 1):
                used_squares.add(path[i])
            
            paths.append(path)

        return sorted(paths, key=len)
    
    # Generate list of moves from the paths we selected
    def ordered_moves(self, paths, max_step):

        moves = []

        for path in paths:
            
            i = min(len(path)-2, max_step)

            while i >= 0:
                
                # Next block is to the right
                if path[i+1][1] > path[i][1]:
                    move = (path[i], 1)
                    if move not in moves:
                        moves.append(move)
                # Next block is to the left
                elif path[i+1][1] < path[i][1]:
                    move = (path[i], 3)
                    if move not in moves:
                        moves.append(move)
                # Next block is above
                elif path[i+1][0] < path[i][0]:
                    move = (path[i], 0)
                    if move not in moves:
                        moves.append(move)
                # Next block is below
                else:
                    move = (path[i], 2)
                    if move not in moves:
                        moves.append(move)
                i -= 1

        return moves

    # Copied and adjusted code from world.py
    # Prof said we could do this https://edstem.org/us/courses/28046/discussion/2187454
    def check_endgame(self, my_pos, adv_pos, chess_board):
        
        board_size = len(chess_board)

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

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_pos = my_pos
        p0_r = find(p0_pos)
        p1_r = find(adv_pos)
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        
        # returns 1 if our agent wins and 0 if we lose?
        return True, p0_score, p1_score

    def get_game_ending_moves(self, adv_pos, valid_moves, chess_board):

        winning_moves = []
        suicide_moves = []
        tie_moves     = []
        remaining     = []

        for move in valid_moves:
            self.simulate_board(move, chess_board, sim=True)
            game_done, p0_score, p1_score = self.check_endgame(move[0], adv_pos, chess_board)
            self.simulate_board(move, chess_board, sim=True)

            if game_done and p0_score > p1_score:
                winning_moves.append(move)
            elif game_done and p1_score > p0_score:
                suicide_moves.append(move)
            elif game_done:
                tie_moves.append(move)
            else:
                remaining.append(move)

        return winning_moves, suicide_moves, tie_moves, remaining
            

    def minimax(self, my_pos, adv_pos, chess_board, alpha, beta, max_player, max_step, depth, max_depth=3):

        # Terminal Node
        if depth == max_depth:
            
            print("Terminal node reached")
            value_of_board = 0
            if max_player:
                value_of_board = self.utility_of_state(my_pos, adv_pos, chess_board)
            else:
                value_of_board = self.utility_of_state(adv_pos, my_pos, chess_board)

            return None, value_of_board

        if max_player:
            
            max_move = None
            path_ls = self.k_shortest_paths(chess_board, my_pos, adv_pos, 5, max_step)
            valid_move_ls = self.ordered_moves(path_ls, max_step)
            winning_moves, suicide_moves, tie_moves, remaining_moves = self.get_game_ending_moves(adv_pos, valid_move_ls, chess_board)

            if len(path_ls) == 0:
                game_done, p0_score, p1_score = self.check_endgame(my_pos, adv_pos, chess_board)
                if game_done and p0_score > p1_score:
                    return None, 1000
                if game_done and p1_score > p0_score:
                    return None, -1000
                
                return None, 0 

            if len(winning_moves) != 0:
                return winning_moves[0], 1000
            
            max_val = -math.inf
            for move in remaining_moves:
                self.simulate_board(move, chess_board, sim=True)
                throwaway, value = self.minimax(adv_pos, move[0], chess_board, alpha, beta, False, max_step, depth+1, max_depth)
                self.simulate_board(move, chess_board, sim=False)
                if value > max_val:
                    max_move = move
                max_val = max(max_val, value)
                alpha = max(alpha, max_val)
                if beta <= alpha:
                    # Prune
                    break
            
            if max_val < 0 and len(tie_moves) != 0:
                return tie_moves[0], 0

            if len(suicide_moves) == len(valid_move_ls):
                # Oops, we lost
                return suicide_moves[0], -1000

            return max_move, max_val
        # Minimum player
        else:
            
            path_ls = self.k_shortest_paths(chess_board, my_pos, adv_pos, 5, max_step)                
            valid_move_ls = self.ordered_moves(path_ls, max_step)
            winning_moves, suicide_moves, tie_moves, remaining_moves = self.get_game_ending_moves(adv_pos, valid_move_ls, chess_board)
            
            if len(winning_moves) != 0:
                return winning_moves[0], -1000
            
            min_val = math.inf

            for move in remaining_moves:
                self.simulate_board(move, chess_board, sim=True)
                throwaway, value = self.minimax(adv_pos, move[0], chess_board, alpha, beta, True, max_step, depth+1, max_depth)
                self.simulate_board(move, chess_board, sim=False)
                min_val = min(min_val, value)
                beta = min(beta, min_val)
                if beta <= alpha:
                    # Prune
                    break

            if min_val > 0 and len(tie_moves) != 0:
                return tie_moves[0], 0

            return None, min_val

    # Return list of valid next positions which can be reached by 1 step
    def valid_next_position(self, state, allow_opp):

        # Retrieve game state at current point of search    
        position, opponent_position, chess_board = state

        board_size = len(chess_board)
        next_pos_ls = []

        # Move Up
        # Check whether leaving upper border of map or if there is a barrier above block
        if position[0] - 1 >= 0 and not chess_board[position[0]][position[1]][0]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0] - 1, position[1]) != opponent_position:
                next_pos_ls.append((position[0] - 1, position[1]))

        # Move Down
        # Check whether leaving lower border of map or if there is a barrier below block
        if position[0] + 1 < board_size  and not chess_board[position[0]][position[1]][2]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0] + 1, position[1]) != opponent_position:
                next_pos_ls.append((position[0] + 1, position[1]))

        # Move Right
        # Check whether leaving right border of map or if there is a barrier to the right
        if position[1] + 1 < board_size and not chess_board[position[0]][position[1]][1]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0], position[1] + 1) != opponent_position:
                next_pos_ls.append((position[0], position[1] + 1))
        
        # Move Left
        # Check whether leaving left border of map or if there is a barrier to the left
        if position[1] - 1 >= 0 and not chess_board[position[0]][position[1]][3]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0], position[1] - 1) != opponent_position:
                next_pos_ls.append((position[0], position[1] - 1))

        return next_pos_ls

    def simulate_board(self, move, chess_board, sim):

        board_size = len(chess_board)

        up = 0
        right = 1
        down = 2
        left = 3

        pos = move[0]
        dir = move[1]

        # Wall Up
        if dir == up:
            if sim:
                chess_board[pos[0]][pos[1]][up] = 1
                if pos[0] != 0:
                    chess_board[pos[0]-1][pos[1]][down] = 1
            else:
                chess_board[pos[0]][pos[1]][up] = 0
                if pos[0] != 0:
                    chess_board[pos[0]-1][pos[1]][down] = 0

        if dir == right:
            if sim:
                chess_board[pos[0]][pos[1]][right] = 1
                if pos[1] != board_size - 1:
                    chess_board[pos[0]][pos[1] + 1][left] = 1
            else:
                chess_board[pos[0]][pos[1]][right] = 0
                if pos[1] != board_size - 1:
                    chess_board[pos[0]][pos[1] + 1][left] = 0

        if dir == down:
            if sim:
                chess_board[pos[0]][pos[1]][down] = 1
                if pos[0] != board_size - 1:
                    chess_board[pos[0] + 1][pos[1]][up] = 1
            else:
                chess_board[pos[0]][pos[1]][down] = 0
                if pos[0] != board_size - 1:
                    chess_board[pos[0] + 1][pos[1]][up] = 0

        if dir == left:
            if sim:
                chess_board[pos[0]][pos[1]][left] = 1
                if pos[1] != 0:
                    chess_board[pos[0]][pos[1] - 1][right] = 1
            else:
                chess_board[pos[0]][pos[1]][left] = 0
                if pos[1] != 0:
                    chess_board[pos[0]][pos[1] - 1][right] = 0                                                

    
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
        start_time = time.time()

        # Run minimax with alpha beta pruning 
        max_move, max_val = self.minimax(my_pos, adv_pos, chess_board, -math.inf, math.inf, True, max_step, 1, 5)
        print("--- %s seconds ---" % (time.time() - start_time))

        return max_move[0], max_move[1]
