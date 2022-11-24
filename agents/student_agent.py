# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
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
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def allPossibleNextMoves(self, chess_board, my_pos, adv_pos, max_step):
        # Set to keep track of previously visited tiles
        alreadyVisitedPosition = set()
        allMoves = set()

        def dfs(position, depth):
            x, y = position
            size = len(chess_board)

            # Do not consider spot opponent is in
            if position == adv_pos:
                return

            # Check if off the board
            if x >= size or x < 0 or y >= size or y < 0:
                return

            # Check if already visited this tile
            if position in alreadyVisitedPosition:
                return
            else:
                alreadyVisitedPosition.add(position)

            if depth < max_step:
                # Check if up barrier in the way
                if chess_board[x][y][self.dir_map["u"]] == False:
                    # If not explore that direction
                    allMoves.add(((x, y), self.dir_map["u"]))
                    dfs((position[0] - 1, position[1]), depth + 1)
                if chess_board[x][y][self.dir_map["d"]] == False:
                    allMoves.add(((x, y), self.dir_map["d"]))
                    dfs((position[0] + 1, position[1]), depth + 1)
                if chess_board[x][y][self.dir_map["r"]] == False:
                    allMoves.add(((x, y), self.dir_map["r"]))
                    dfs((position[0], position[1] + 1), depth + 1)
                if chess_board[x][y][self.dir_map["l"]] == False:
                    allMoves.add(((x, y), self.dir_map["l"]))
                    dfs((position[0], position[1] - 1), depth + 1)

        dfs(my_pos, 0)
        return allMoves

    def distanceToOpponent(self, my_pos, adv_pos):
        return adv_pos[0][0] - my_pos[0][0] + adv_pos[0][1] - my_pos[0][1]

    def heuristic(self, chess_board, adv_pos, new_move):
        # If you can win on the move return - 10000

        return math.dist(new_move[0], adv_pos)

    # function for finding the best move based on the heuristic
    def findBestMove(self, allMoves, chess_board, my_pos, adv_pos, max_step):

        min_H_move = allMoves.pop()
        min_H = self.heuristic(chess_board, adv_pos, min_H_move)

        for new_move in allMoves:
            new_H = self.heuristic(chess_board, adv_pos, new_move)
            if (new_H < min_H):
                min_H = new_H
                min_H_move = new_move

        return min_H_move

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
        # dummy return

        allMoves = self.allPossibleNextMoves(
            chess_board, my_pos, adv_pos, max_step)

        bestMove = self.findBestMove(
            allMoves, chess_board, my_pos, adv_pos, max_step)

        return bestMove
