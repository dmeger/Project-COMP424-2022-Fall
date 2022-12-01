# Student agent: Add your own agent here
import numpy as np
from copy import copy
from agents.agent import Agent
from store import register_agent
from collections import deque, defaultdict
from queue import PriorityQueue

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PriorityQueueItem:
    priority: int
    item: Any = field(compare=False)


@register_agent("student_agent2")
class StudentAgent2(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent2, self).__init__()
        self.name = "StudentAgent2"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.chess_board = np.array(np.array([]))
        self.my_allowed_moves = None
        self.allowed_spots = None
        self.adv_allowed_moves = None
        self.my_losing_moves = None
        self.adv_losing_moves = None
        self.my_winning_moves = None
        self.adv_winning_moves = None
        self.count = 0
        self.max_steps = 10
        self.move_num = 0
        self.squares = None
        self.adv_pos = None

    def mutate_board(self, x, y, i, set_to):
        """
        add a barrier to the board temporarily to see its effect
        """
        self.chess_board[x, y, i] = set_to
        if i == self.dir_map["u"] and x > 0:
            self.chess_board[x - 1, y, self.dir_map["d"]] = set_to
        elif i == self.dir_map["r"] and y < len(self.chess_board[0]) - 1:
            self.chess_board[x, y + 1, self.dir_map["l"]] = set_to
        elif i == self.dir_map["d"] and x < len(self.chess_board) - 1:
            self.chess_board[x + 1, y, self.dir_map["u"]] = set_to
        elif i == self.dir_map["l"] and y > 0:
            self.chess_board[x, y - 1, self.dir_map["r"]] = set_to

    def dist_to_squares(self, pos, adv_pos, max_depth=8):
        """
        get the distance to all squares on the board
        (depth limited to reduce runtime)
        """

        fx, fy = pos
        ax, ay = adv_pos
        cache = set()
        dist_to = np.ones([self.chess_board.shape[0], self.chess_board.shape[1]], dtype=int) * (
                2 * self.chess_board.shape[1] + 2)
        queue = deque()
        queue.appendleft((fx, fy, 0))
        while len(queue) > 0:
            fx, fy, steps = queue.pop()
            cache.add((fx, fy))
            dist_to[fx][fy] = min(steps, dist_to[fx][fy])
            if steps >= max_depth:
                return dist_to
            for j in range(4):
                if not self.chess_board[fx][fy][j]:
                    if j % 2 == 0:
                        if (fx - (1 - j), fy) not in cache and fx - (1 - j) >= 0 and fy >= 0 and fx - (
                                1 - j) < len(self.chess_board) and fy < len(self.chess_board[0]) and (
                                fx - (1 - j) != ax or fy != ay
                        ):
                            queue.appendleft((fx - (1 - j), fy, steps + 1))
                    elif (fx, fy + (2 - j)) not in cache and fx >= 0 and fy + (2 - j) >= 0 and fx < len(
                            self.chess_board) and fy + (2 - j) < len(self.chess_board[0]) and (
                            fx != ax or fy + (2 - j) != ay
                    ):
                        queue.appendleft((fx, fy + (2 - j), steps + 1))
        return dist_to

    def get_utility(self, move, adv_pos):
        """
        get the value of a given move based on the distance to all the square
        """

        x, y, i = move
        self.mutate_board(x, y, i, set_to=True)
        adv_dist_to = self.dist_to_squares(adv_pos, (x, y))
        my_dist_to = self.dist_to_squares((x, y), adv_pos)
        controlling = adv_dist_to - my_dist_to
        utility = (controlling, controlling.sum())
        self.mutate_board(x, y, i, set_to=False)
        return utility

    def a_star(self, my_pos, target_pos, block_adv=False, limit=0):
        """
        a-star search a shortest path from pos0 to pos1
        """

        fx, fy = my_pos
        ax, ay = target_pos
        adv_x, adv_y = self.adv_pos

        best, best_priority = [], float('inf')
        cache = set()
        queue = PriorityQueue()
        queue.put(PriorityQueueItem(0, ((fx, fy), [(fx, fy)])))
        while not queue.empty():
            item = queue.get()
            priority = item.priority
            (fx, fy), path = item.item
            cache.add((fx, fy))

            if priority < best_priority:
                best_priority = priority
                best = path

            if 0 < limit <= len(path):
                return best

            for j in range(4):
                if not self.chess_board[fx][fy][j]:
                    successor_path = copy(path)

                    if j % 2 == 0:
                        if (fx - (1 - j), fy) not in cache and fx - (1 - j) >= 0 and fy >= 0 and fx - (
                                1 - j) < len(self.chess_board) and fy < len(self.chess_board[0]):
                            step_x, step_y = fx - (1 - j), fy
                            if block_adv and step_x == adv_x and step_y == adv_y:
                                continue
                            elif step_x == ax and step_y == ay:
                                return path
                            successor = (step_x, step_y)
                            successor_path.append(successor)
                            queue.put(
                                PriorityQueueItem(abs(step_x - ax) + abs(step_y - ay), (successor, successor_path)))

                    elif (fx, fy + (2 - j)) not in cache and fx >= 0 and fy + (2 - j) >= 0 and fx < len(
                            self.chess_board) and fy + (2 - j) < len(self.chess_board[0]):
                        step_x, step_y = fx, fy + (2 - j)
                        if block_adv and step_x == adv_x and step_y == adv_y:
                            continue
                        elif step_x == ax and step_y == ay:
                            return path
                        successor = (step_x, step_y)
                        successor_path.append(successor)
                        queue.put(PriorityQueueItem(abs(step_x - ax) + abs(step_y - ay), (successor, successor_path)))

        # print("NO PATH EXISTS!")
        return best

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

    def get_game_ending_moves(self, moves, adv_pos):
        """
        returns all the moves that will win/lose the game from a list of valid moves
        """

        winning_moves = set()
        losing_moves = set()

        for move in moves:
            x, y, i = move
            self.mutate_board(x, y, i, set_to=True)
            over, my_score, adv_score = self.check_endgame((x, y), adv_pos, self.chess_board)
            if over:
                if my_score > adv_score:
                    winning_moves.add(move)
                else:
                    losing_moves.add(move)
            self.mutate_board(x, y, i, set_to=False)

        return winning_moves, losing_moves

    def get_allowed_moves(self, my_pos, adv_pos, max_step, path):
        """
        returns a set of all valid moves we can take from a given subset of squares
        """

        adv_x, adv_y = adv_pos
        can_place = set()
        spot_dict = defaultdict(list)
        visited = set()
        path_set = set(path)

        def place_wall(x, y, steps_left):
            if (x, y) in visited or (x, y) not in path_set:
                return

            if steps_left < 0 or x < 0 or y < 0 or \
                    x >= len(self.chess_board) or y >= len(self.chess_board[0]) \
                    or (adv_x == x and adv_y == y):
                return

            visited.add((x, y))
            for i in range(4):
                if not self.chess_board[x][y][i]:
                    can_place.add((x, y, i))
                    spot_dict[(x, y)].append((x, y, i))
                    if i % 2 == 0:
                        place_wall(x - (1 - i), y, steps_left - 1)
                    else:
                        place_wall(x, y + (2 - i), steps_left - 1)

        my_x, my_y = my_pos
        place_wall(my_x, my_y, max_step)
        return can_place, spot_dict

    def get_move(self, my_pos, adv_pos, max_step, utilities):
        """
        given a set of moves take the one with maximum utility
        (should in theory be the place to return the best move)
        """

        for move in self.my_winning_moves:
            x, y, i = move
            return (x, y), i

        while not utilities.empty():
            x, y, i = utilities.get().item
            if (x, y, i) not in self.my_losing_moves:
                self.mutate_board(x, y, i, set_to=True)
                shortest_path = self.a_star(adv_pos, (x, y))
                self.adv_allowed_moves, adv_allowed_spots = self.get_allowed_moves(adv_pos, (x, y), max_step,
                                                                                   shortest_path)
                adv_path_moves, adv_utilities = self.get_path_moves(shortest_path, (x, y), adv_allowed_spots)
                self.adv_winning_moves, self.adv_losing_moves = self.get_game_ending_moves(adv_path_moves,
                                                                                           (x, y))
                self.mutate_board(x, y, i, set_to=False)

                if len(self.adv_winning_moves) == 0:
                    return (x, y), i

        # print("No utility moves available")

        tmp = None
        for move in self.my_allowed_moves:
            x, y, i = move
            filled_block = 0
            for j in range(4):
                if self.chess_board[x][y][j]:
                    filled_block += 1
            if (x, y, i) not in self.my_losing_moves:
                if filled_block < 2:
                    return (x, y), i
                else:
                    tmp = (x, y), i

        if tmp is None:
            print("Could not find valid move")
            if len(self.my_losing_moves) > 0:
                (x, y, i) = next(iter(self.my_losing_moves))
                tmp = (x, y), i

        return tmp

    def get_path_moves(self, shortest_path, adv_pos, allowed_spots):
        shortest_path_moves = []
        utilities = PriorityQueue()
        for spot in shortest_path[::-1]:
            if spot in allowed_spots:
                for move in allowed_spots.get(spot):
                    shortest_path_moves.append(move)
                    u_map, u_score = self.get_utility(move, adv_pos)
                    utilities.put(PriorityQueueItem(-u_score, move))

        return shortest_path_moves, utilities

    def build_tree(self, my_pos, adv_pos, max_step, moves, winning_moves=0, depth=0):
        """
        Builds an entire search tree of possible moves to take
        (Very long runtime!)
        """

        if depth > 20:
            return moves, winning_moves

        self.my_allowed_moves, allowed_spots = self.get_allowed_moves(my_pos, adv_pos, max_step, self.squares)
        self.my_winning_moves, self.my_losing_moves = self.get_game_ending_moves(self.my_allowed_moves, adv_pos)
        if len(self.my_winning_moves) > 0:
            return moves, 1

        t_winning_moves = float('-inf')
        t_moves = None
        for j, move in enumerate(self.my_allowed_moves):
            x, y, i = move
            if move in self.my_losing_moves:
                continue

            moves.append(move)

            self.mutate_board(x, y, i, set_to=True)
            self.adv_allowed_moves, allowed_spots = self.get_allowed_moves(adv_pos, (x, y), max_step, self.squares)
            self.adv_winning_moves, self.adv_losing_moves = self.get_game_ending_moves(self.adv_allowed_moves, (x, y))
            if len(self.adv_winning_moves) > 0:
                moves.pop()
                continue

            m_winning_moves = float('inf')
            m_moves = None
            for k, adv_move in enumerate(self.adv_allowed_moves):
                ax, ay, ai = adv_move
                if adv_move in self.adv_losing_moves:
                    continue

                self.mutate_board(ax, ay, ai, set_to=True)
                self.count += 1
                d_moves, d_winning_moves = self.build_tree((x, y), (ax, ay), max_step, list(moves), winning_moves,
                                                           depth + 1)
                self.mutate_board(ax, ay, ai, set_to=False)

                if d_winning_moves < m_winning_moves:
                    m_moves = d_moves
                    m_winning_moves = d_winning_moves

            self.mutate_board(x, y, i, set_to=False)

            if m_winning_moves > t_winning_moves:
                t_moves = m_moves
                t_winning_moves = m_winning_moves
            elif m_moves is None:
                return moves, 1

            moves.pop()

        if t_moves:
            return t_moves, t_winning_moves
        else:
            return moves, 1

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
        self.chess_board = chess_board
        self.max_steps = max_step
        self.count = 0
        self.adv_pos = adv_pos

        '''
        if self.move_num == 0 and chess_board.shape[0] <= 7:
            self.squares = [(i, j) for j in range(self.chess_board.shape[0]) for i in range(self.chess_board.shape[0])]
            tree, n = self.build_tree(my_pos, adv_pos, max_step, [])
            # refresh state
            self.chess_board = chess_board
            self.max_steps = max_step
            self.my_allowed_moves = None
            self.allowed_spots = None
            self.adv_allowed_moves = None
            self.my_losing_moves = None
            self.adv_losing_moves = None
            self.my_winning_moves = None
            self.adv_winning_moves = None
        '''
        self.move_num += 1

        shortest_path = self.a_star(my_pos, adv_pos)
        self.my_allowed_moves, self.allowed_spots = self.get_allowed_moves(my_pos, adv_pos, max_step, shortest_path)

        shortest_path_moves, utilities = self.get_path_moves(shortest_path, adv_pos, self.allowed_spots)

        try:
            if not utilities.empty():
                item = utilities.get()
                best_score, best_move = item.priority, item.item
                if best_score > 0:
                    opp_of_board = (len(chess_board) - my_pos[0] - 1, len(chess_board) - my_pos[1] - 1)
                    opp_path = self.a_star(my_pos, opp_of_board, block_adv=True, limit=20)
                    """
                    dists = self.dist_to_squares(my_pos, adv_pos, max_depth=8)
                    index = np.where(dists <= 2 * self.chess_board.shape[1], dists, 0).argmax()
                    index_x, index_y = index // self.chess_board.shape[1], index % self.chess_board.shape[1]
                    opp_path = self.a_star(my_pos, (index_x, index_y), block_adv=True)
                    """
                    allowed_moves, allowed_spots = self.get_allowed_moves(my_pos, adv_pos, max_step,
                                                                          opp_path)
                    middle_path_moves, m_utilities = self.get_path_moves(opp_path, adv_pos, allowed_spots)
                    while not m_utilities.empty():
                        utilities.put(m_utilities.get())
                utilities.put(item)
        except:
            pass

        self.my_winning_moves, self.my_losing_moves = self.get_game_ending_moves(shortest_path_moves, adv_pos)
        return self.get_move(my_pos, adv_pos, max_step, utilities)