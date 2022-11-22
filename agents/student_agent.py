# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import copy

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
        self.game_state = None

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
        # save current game state
        self.game_state = (my_pos, adv_pos, chess_board)

        return my_pos, self.dir_map["u"]

    def distance_to_position(self, game_state, max_dfs_depth=8):
  
        position, opponent_position, chess_board = game_state
        distances = np.zeros(self.board_size, self.board_size)
        
        queue = [(position, 0)]
        encountered = [position]
        while len(queue) > 0:

            cur_position, distance = queue.pop()
            distances[cur_position] = distance

            if distance >= max_dfs_depth:
                continue

            for next_position in self.valid_next_position((cur_position, opponent_position, chess_board), False):
                
                if next_position not in encountered:
                    queue.append([next_position, distance + 1])
                    encountered.append(next_position)
        
        return distances
    
    # Return the heuristic value of a given move
    def move_utility(self, move, adv_pos):

        chess_board = self.game_state[2]

        # Simulate the move
        self.simulate_board(move, set=True)

        dist     = self.distance_to_position((move[0], adv_pos, chess_board))
        opp_dist = self.distance_to_position((adv_pos, move[0], chess_board))

        # Reset the board
        self.simulate_board(move, set=False)

        diff = dist - opp_dist

        return np.sum(diff > 0)

    # Return list of valid next positions which can be reached by 1 step
    def valid_next_position(self, state, allow_opp):

        # Retrieve game state at current point of search    
        position, opponent_position, chess_board = state

        next_pos_ls = []

        # Move Up
        # Check whether leaving upper border of map or if there is a barrier above block
        if position[1] - 1 >= 0 and not chess_board[position[0]][position[1]][0]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0], position[1] - 1) != opponent_position:
                next_pos_ls.append((position[0], position[1] - 1))

        # Move Down
        # Check whether leaving lower border of map or if there is a barrier below block
        if position[1] + 1 < self.board_size  and not chess_board[position[0]][position[1]][2]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0], position[1] + 1) != opponent_position:
                next_pos_ls.append((position[0], position[1] + 1))

        # Move Right
        # Check whether leaving right border of map or if there is a barrier to the right
        if position[0] + 1 < self.board_size and not chess_board[position[0]][position[1]][1]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0] + 1, position[1]) != opponent_position:
                next_pos_ls.append((position[0] + 1, position[1]))
        
        # Move Left
        # Check whether leaving left border of map or if there is a barrier to the left
        if position[1] - 1 >= 0 and not chess_board[position[0]][position[1]][3]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (position[0] - 1, position[1]) != opponent_position:
                next_pos_ls.append((position[0] - 1, position[1]))

        return next_pos_ls

    # Determine the 5 shortest paths from the current position to the target position
    # TODO: currently only returns shortest path
    def shortest_path(self, game_state):

        # Parse current game state
        cur_position, target_position, chess_board = game_state

        start = A_Star_Node(cur_position)
        start.g = 0
        start.h = self.a_star_heuristic(cur_position, target_position)
        end     = A_Star_Node(target_position)

        # List of paths to return
        path_ls = []
        
        queue = []
        explored  = []
        queue.append((start, start.g + start.h))

        while len(queue) > 0 and len(path_ls) < 1:

            # Retrieve the lowest f = g + h value node from the queue
            current_node = min(queue, key = lambda t: t[1])
            
            # Remove the node from the queue
            queue.remove(current_node)
            
            # Mark the node as explored
            explored.append(current_node.pos)

            # Reached the goal
            if current_node.pos == end.pos:
                path = [current_node.pos]

                # Backtrack through the path till the start point has been reached
                while current_node.parent != None:
                    path.append(current_node.parent.pos)
                    current_node = current_node.parent
                    
                # Add the configured path to the list of paths
                path_ls.append(path[::-1])
                continue
        
            # Iterate through the neighbours of the current node
            for child in self.valid_next_position((current_node.pos, target_position, chess_board), True):

                if child not in explored:

                    child_node = A_Star_Node(child, current_node.pos)
                    child_node.g = current_node.g + 1
                    child_node.h = self.a_star_heuristic(child_node.pos, target_position)
                
                    queue.append((child_node, child_node.g + child_node.h))
    
    # Generate possible game moves from a given path
    def moves_from_path(self, path, max_step):
        
        move_ls = []
        i = 0

        # Each step up to maximum steps in the path generates a possible move
        # Note we cannot move onto the square the opponent is on
        while i < max_step and i != len(path) - 1:
            
            # Next block is to the right
            if path[i+1][0] > path[i][0]:
                move_ls.append((path[i], 1))
            # Next block is to the left
            elif path[i+1][0] < path[i][0]:
                move_ls.append((path[i], 3))
            # Next block is above
            elif path[i+1][1] < path[i][1]:
                move_ls.append((path[i], 0))
            # Next block is below
            else:
                move_ls.append((path[i], 2))

            i += 1

        return move_ls

    def simulate_board(self, move, set):

        chess_board = self.game_state[2]
        pos = move[0]
        dir = move[1]

        if set:
            chess_board[pos[0]][pos[1]][dir] = 1
        else:
            chess_board[pos[0]][pos[1]][dir] = 0
