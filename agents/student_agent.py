# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import sys
import numpy as np
import copy
import math
import time
import heapq


class Priority_Queue:

    def __init__(self, item_ls):
        self.h = []
        for item in item_ls:
            heapq.heappush(self.h, item)

    def is_empty(self):
        return len(self.h) == 0

    def push(self, value, item):
        heapq.heappush(self.h, (value, item))

    def pop(self):
        if not self.is_empty():
            return heapq.heappop(self.h)
        return None

    def get_top(self):
        if not self.is_empty():
            return self.h[0]
        return None, None


class A_Star_Node():

    def __init__(self, pos, parent=None):

        self.pos = pos
        self.parent = parent
        self.g = 0
        self.h = 0


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
        self.move_num = 0
        self.game_time = 0

    # Distance from current position to all squares on the chess_board
    def distance_to_position(self, my_pos, adv_pos, chess_board, max_dfs_depth=14):

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

            for next_position in self.valid_next_position(cur_position, adv_pos, chess_board, False):

                if next_position not in encountered:
                    queue.append([next_position, distance + 1])
                    encountered.add(next_position)

        return distances

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

    # Return the heuristic value of a given move
    def utility_of_state(self, my_pos, adv_pos, chess_board):

        dist = self.distance_to_position(my_pos, adv_pos, chess_board)
        opp_dist = self.distance_to_position(adv_pos, my_pos, chess_board)

        diff = dist - opp_dist

        return np.sum(diff > 0) - np.sum(diff < 0)

    def a_star_heuristic(self, my_pos, adv_pos):
        return abs(my_pos[0]-adv_pos[0]) + abs(my_pos[1]-adv_pos[1])

    # find shortest path from my position to target position
    def a_star(self, chess_board, my_pos, target_pos, used_squares, max_step, allow_opp=True, adv_pos=None):

        start = A_Star_Node(my_pos)
        start.g = 0
        start.h = self.a_star_heuristic(my_pos, target_pos)
        end = A_Star_Node(target_pos)

        queue = set()
        explored = set()
        queue.add((start, start.g + start.h))

        while len(queue) > 0:

            # Retrieve the lowest f = g + h value node from the queue
            current_node = min(queue, key=lambda t: t[1])

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

                    if not (current_a_star_node.parent.pos in used_squares) and current_a_star_node.parent.parent != None:
                        unique_square = True

                    path.append(current_a_star_node.parent.pos)
                    current_a_star_node = current_a_star_node.parent

                # Add the configured path to the list of paths
                if not unique_square:
                    return None

                return path[::-1]

            # Retrieve the neighbours of the current node
            next_positions = self.valid_next_position(
                current_a_star_node.pos, adv_pos, chess_board, allow_opp)

            queued = False
            for child in next_positions:

                if child not in explored and (child not in used_squares or current_a_star_node.g + 1 > max_step):

                    queued = True
                    child_node = A_Star_Node(child, current_a_star_node)
                    child_node.g = current_a_star_node.g + 1
                    child_node.h = self.a_star_heuristic(
                        child_node.pos, target_pos)
                    queue.add((child_node, child_node.g + child_node.h))

            if not queued:

                for child in next_positions:

                    if child not in explored:
                        child_node = A_Star_Node(child, current_a_star_node)
                        child_node.g = current_a_star_node.g + 1
                        child_node.h = self.a_star_heuristic(
                            child_node.pos, target_pos)
                        queue.add((child_node, child_node.g + child_node.h))

        return None

    # Retrieve the k shortest paths from my position to target position
    def k_shortest_paths(self, chess_board, my_pos, target_pos, k, max_step, allow_opp=True, adv_pos=None):

        paths = []
        used_squares = set()

        while len(paths) < k:

            path = self.a_star(chess_board, my_pos, target_pos,
                               used_squares, max_step, allow_opp, adv_pos)

            if path == None:
                break

            for i in range(1, len(path) - 1):
                used_squares.add(path[i])

            paths.append(path)

        return sorted(paths, key=len)

    # Generate list of moves from the paths we selected
    def get_moves(self, paths, max_step, chess_board):

        optimal_moves = set()
        all_moves = set()

        for path in paths:

            i = min(len(path)-2, max_step)

            while i >= 0:

                # Next block is to the right
                if path[i+1][1] > path[i][1]:
                    move = (path[i], 1)
                    optimal_moves.add(move)
                # Next block is to the left
                elif path[i+1][1] < path[i][1]:
                    move = (path[i], 3)
                    optimal_moves.add(move)
                # Next block is above
                elif path[i+1][0] < path[i][0]:
                    move = (path[i], 0)
                    optimal_moves.add(move)
                # Next block is below
                else:
                    move = (path[i], 2)
                    optimal_moves.add(move)

                for j in range(4):
                    if not chess_board[path[i][0]][path[i][1]][j]:
                        move = (path[i], j)
                        all_moves.add(move)

                i -= 1

        return all_moves, optimal_moves

    def random_move(self, chess_board, my_pos, adv_pos, max_step):

        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    # Return list of valid next positions which can be reached by 1 step
    def valid_next_position(self, my_pos, adv_pos, chess_board, allow_opp):

        board_size = len(chess_board)
        next_pos_ls = []

        # Move Up
        # Check whether leaving upper border of map or if there is a barrier above block
        if my_pos[0] - 1 >= 0 and not chess_board[my_pos[0]][my_pos[1]][0]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (my_pos[0] - 1, my_pos[1]) != adv_pos:
                next_pos_ls.append((my_pos[0] - 1, my_pos[1]))

        # Move Down
        # Check whether leaving lower border of map or if there is a barrier below block
        if my_pos[0] + 1 < board_size and not chess_board[my_pos[0]][my_pos[1]][2]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (my_pos[0] + 1, my_pos[1]) != adv_pos:
                next_pos_ls.append((my_pos[0] + 1, my_pos[1]))

        # Move Right
        # Check whether leaving right border of map or if there is a barrier to the right
        if my_pos[1] + 1 < board_size and not chess_board[my_pos[0]][my_pos[1]][1]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (my_pos[0], my_pos[1] + 1) != adv_pos:
                next_pos_ls.append((my_pos[0], my_pos[1] + 1))

        # Move Left
        # Check whether leaving left border of map or if there is a barrier to the left
        if my_pos[1] - 1 >= 0 and not chess_board[my_pos[0]][my_pos[1]][3]:

            # If we are not searching for opponent the opponent acts as a wall
            if allow_opp or (my_pos[0], my_pos[1] - 1) != adv_pos:
                next_pos_ls.append((my_pos[0], my_pos[1] - 1))

        return next_pos_ls

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

    # Determine the Game Ending qualities of given moves
    def get_game_ending_moves(self, adv_pos, valid_moves, chess_board):

        winning_moves = []
        suicide_moves = []
        tie_moves = []
        remaining = []

        for move in valid_moves:
            self.simulate_board(move, chess_board, sim=True)
            game_done, p0_score, p1_score = self.check_endgame(
                move[0], adv_pos, chess_board)

            if game_done and p0_score > p1_score:
                winning_moves.append(move)
            elif game_done and p1_score > p0_score:
                suicide_moves.append(move)
            elif game_done:
                tie_moves.append(move)
            else:
                remaining.append(move)
                #score = self.utility_of_state(move[0], adv_pos, chess_board)
                #remaining.append((-score, move))

            self.simulate_board(move, chess_board, sim=False)

        return winning_moves, suicide_moves, tie_moves, remaining

    def will_it_lose_me_the_game(self, move, adv_pos, chess_board, max_step):

        my_pos, dir = move
        self.simulate_board(move, chess_board, sim=True)
        adv_path_ls = self.k_shortest_paths(
            chess_board, adv_pos, move[0], 3, max_step)
        allowed_adv_moves, optimal_adv_moves = self.get_moves(
            adv_path_ls, max_step, chess_board)

        adv_winning_moves, adv_losing_moves, tie_moves, other_moves = self.get_game_ending_moves(
            my_pos, allowed_adv_moves, chess_board)
        self.simulate_board(move, chess_board, sim=False)

        if len(adv_winning_moves) != 0:
            return 1, 0

        return 0, len(adv_losing_moves)

    def best_random_move(self, my_pos, adv_pos, chess_board, max_step):

        # Generate random move
        best_val = -math.inf
        best_move = None
        for i in range(50):
            random_move = self.random_move(
                chess_board, my_pos, adv_pos, max_step)
            self.simulate_board(random_move, chess_board, sim=True)
            game_end, p0_score, p1_score = self.check_endgame(
                my_pos, adv_pos, chess_board)

            if not game_end:
                val = self.utility_of_state(my_pos, adv_pos, chess_board)
                if val > best_val:
                    best_move = random_move
                    best_val = val
            elif p0_score > p1_score:
                return 1000, random_move
            elif p0_score == p1_score:
                if -500 > best_val:
                    best_move = random_move
                    best_val = -500
            else:
                if -1000 > best_val:
                    best_move = random_move
                    best_val = -1000
            self.simulate_board(random_move, chess_board, sim=False)

        return best_val, best_move

    def get_best_move(self, my_pos, adv_pos, chess_board, max_step):

        path_ls = self.k_shortest_paths(
            chess_board, my_pos, adv_pos, 3, max_step)
        allowed_moves, optimal_moves = self.get_moves(
            path_ls, max_step, chess_board)

        winning_moves, losing_moves, tie_moves, other_moves = self.get_game_ending_moves(
            adv_pos, allowed_moves, chess_board)

        if len(winning_moves) != 0:
            return winning_moves[0]

        value_queue = Priority_Queue([])
        for move in other_moves:
            self.simulate_board(move, chess_board, sim=True)
            value = self.utility_of_state(my_pos, adv_pos, chess_board)
            value_queue.push(-value, move)
            self.simulate_board(move, chess_board, sim=False)

        if value_queue.is_empty():
            random_val, random_move = self.best_random_move(
                my_pos, adv_pos, chess_board, max_step)
            if random_val <= -1000 and len(tie_moves) != 0:
                return tie_moves[0]

            return random_move

        top_val, top_move = value_queue.pop()

        if top_val != None:
            top_val = -top_val

        if top_val < 0:

            # Want to run away
            target_pos = ((len(chess_board) - my_pos[0] - 1) % len(
                chess_board), (len(chess_board) - my_pos[1] - 1) % len(chess_board))
            # Path to runaway with opponent acting as a wall
            runaway_paths = self.k_shortest_paths(
                chess_board, my_pos, target_pos, 1, max_step, allow_opp=False, adv_pos=adv_pos)

            new_moves, new_optimal_moves = self.get_moves(
                runaway_paths, max_step, chess_board)

            new_winning_moves, new_losing_moves, new_tie_moves, new_other_moves = self.get_game_ending_moves(
                adv_pos, new_moves, chess_board)

            if len(new_winning_moves) != 0:
                return new_winning_moves[0]

            for new_move in new_other_moves:
                self.simulate_board(new_move, chess_board, sim=True)
                new_move_val = self.utility_of_state(
                    my_pos, adv_pos, chess_board)
                value_queue.push(-new_move_val, new_move)
                self.simulate_board(new_move, chess_board, sim=False)

        if top_val != None and top_move != None:
            value_queue.push(-top_val, top_move)

        # No good moves to take, try to take a non-losing random step
        if value_queue.is_empty():
            random_val, random_move = self.best_random_move(
                my_pos, adv_pos, chess_board, max_step)
            if random_val <= -1000 and len(tie_moves) != 0:
                return tie_moves[0]
            return random_move

        next_val, next_move = value_queue.pop()
        next_val = - next_val
        final_queue = Priority_Queue([])
        can_lose, bonus = self.will_it_lose_me_the_game(
            next_move, adv_pos, chess_board, max_step)
        if not can_lose:
            final_queue.push(-(next_val + bonus), next_move)

        while not value_queue.is_empty():
            following_val, following_move = value_queue.pop()
            following_val = -following_val
            if following_val != next_val:
                break
            else:
                next_val, next_move = following_val, following_move
                can_lose, bonus = self.will_it_lose_me_the_game(
                    next_move, adv_pos, chess_board, max_step)
                if not can_lose:
                    final_queue.push(-(next_val + bonus), next_move)

        if not final_queue.is_empty():
            best_val, best_move = final_queue.pop()
            return best_move

        random_val, random_move = self.best_random_move(
            my_pos, adv_pos, chess_board, max_step)
        if random_val <= -1000 and len(tie_moves) != 0:
            return tie_moves[0]

        return random_move

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
        self.game_time = time.time()
        max_move = self.get_best_move(my_pos, adv_pos, chess_board, max_step)
        print("--- %s seconds ---" % (time.time() - self.game_time))

        return max_move
