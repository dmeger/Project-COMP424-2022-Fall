# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import math
import copy
import time


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

    def getAllMoves(self, chess_board, my_pos, adv_pos, max_step):
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

    # Copied and adjusted code from world.py
    # Prof said we could do this https://edstem.org/us/courses/28046/discussion/2187454
    def checkEndGame(self, chess_board, my_pos, adv_pos):
        board_size = len(chess_board)
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
        p0_r = find(my_pos)
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

    def setBarrier(self, chess_board, move):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Create a copy of the board so the original stays intact
        newBoard = copy.deepcopy(chess_board)

        # This is all copied from world.py
        pos, dir = move
        x, y = pos
        newBoard[x, y, dir] = True
        move = moves[dir]
        newBoard[x + move[0], y + move[1], opposites[dir]] = True
        return newBoard

    def heuristic(self, chess_board, adv_pos, my_pos):
        # Check if someone can win the game
        someoneWon, myScore, theirScore = self.checkEndGame(
            chess_board, my_pos, adv_pos)
        if myScore > theirScore:
            return -1000 - myScore
        if myScore < theirScore:
            return 1000 + myScore

        # Initialize a heuris
        heurVal = math.dist(my_pos, adv_pos)

        # Count the # of walls that surround the position
        wallCount = 0
        x, y = my_pos
        for wall in chess_board[x][y]:
            if wall == True:
                wallCount = wallCount + 1

        heurVal = heurVal + (wallCount * 10)

        return heurVal

    # Create a mimMax algorithm that returns the best heuristic for a given board
    # If we are the max player, we want to maximize the heuristic
    # If we are the min player, we want to minimize the heuristic
    # If someone wins then we want to return the heuristic
    def minimax(self, chess_board, adv_pos, my_pos, depth, alpha, beta, isMax):
        # Check if someone can win the game
        someoneWon, myScore, theirScore = self.checkEndGame(
            chess_board, my_pos, adv_pos)
        if myScore > theirScore:
            return -1000 - myScore
        if myScore < theirScore:
            return 1000 + myScore

        # If we are at the max depth, return the heuristic
        if depth == 0:
            return self.heuristic(chess_board, adv_pos, my_pos)

        # If we are the max player, we want to maximize the heuristic
        if isMax:
            bestVal = -math.inf
            for move in self.getAllMoves(chess_board, my_pos, adv_pos, 1):
                newBoard = self.setBarrier(chess_board, move)
                bestVal = max(bestVal, self.minimax(
                    newBoard, adv_pos, my_pos, depth - 1, alpha, beta, False))
                alpha = max(alpha, bestVal)
                if beta <= alpha:
                    break
            return bestVal
        # If we are the min player, we want to minimize the heuristic
        else:
            bestVal = math.inf
            for move in self.getAllMoves(chess_board, adv_pos, my_pos, 1):
                newBoard = self.setBarrier(chess_board, move)
                bestVal = min(bestVal, self.minimax(
                    newBoard, adv_pos, my_pos, depth - 1, alpha, beta, True))
                beta = min(beta, bestVal)
                if beta <= alpha:
                    break
            return bestVal

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

        allMoves = self.getAllMoves(chess_board, my_pos, adv_pos, max_step)

        # bestMove = self.findBestMove(
        #     allMoves, chess_board, my_pos, adv_pos, max_step)

        bestMove = False
        minH = 10000

        # Call the minmax function from current board state
        for move in allMoves:
            newBoard = self.setBarrier(chess_board, move)
            h = self.minimax(newBoard, adv_pos, my_pos,
                             5, -math.inf, math.inf, True)
            if h < minH:
                minH = h
                bestMove = move

        return bestMove
