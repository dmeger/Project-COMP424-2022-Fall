from agents.agent import Agent
from store import register_agent
import sys
import math
import copy
import time


@register_agent("shallow_agent")
class ShallowAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    max_depth = 3

    def __init__(self):
        super(ShallowAgent, self).__init__()
        self.name = "ShallowAgent"
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
            if x > size or x < 0 or y > size or y < 0:
                return

            # Check if already visited this tile
            if position in alreadyVisitedPosition:
                return
            else:
                alreadyVisitedPosition.add(position)

            if depth <= max_step:
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
        m = moves[dir]
        newBoard[x + m[0], y + m[1], opposites[dir]] = True
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

        # Count the # of walls that surround adv_pos
        advWallCount = 0
        xAdv, yAdv = adv_pos
        for wall in chess_board[xAdv][yAdv]:
            if wall == True:
                advWallCount = advWallCount + 1

        heurVal = heurVal - (advWallCount * 20)

        return heurVal

    # Create a mimMax algorithm that returns the best heuristic for a given board
    # If we are the max player, we want to maximize the heuristic
    # If we are the min player, we want to minimize the heuristic
    # If someone wins then we want to return the heuristic
    def minimax(self, chess_board, my_pos, adv_pos, depth, whosTurn, max_step):
        # If its my turn I want to return the min heuristic
        if whosTurn == 0:
            bestHeur = math.inf
        # If its the opponents turn I want to return the max heuristic
        else:
            bestHeur = -math.inf

        # Check if someone can win the game
        # If its my turn and I can win, return -1000
        # If its the opponents turn and they can win, return 1000
        notImportant, myScore, theirScore = self.checkEndGame(
            chess_board, my_pos, adv_pos)
        # If I can win and its my turn return -1000
        if myScore > theirScore and whosTurn == 0:
            return -1000 * depth
        # If they can win and its their turn return 1000
        if myScore < theirScore and whosTurn == 1:
            return 1000 * depth

        # If its my turn and I can't win, find the move with the lowest heuristic
        # If its the opponents turn and they can't win, find the move with the highest heuristic
        # If we are at the max depth, return the heuristic
        if depth <= 0:
            newBoard = self.setBarrier(chess_board, move)
            return self.heuristic(newBoard, adv_pos, my_pos)

        # If it is my turn
        if whosTurn == 0:
            # Get all the moves I can make
            allMoves = self.getAllMoves(chess_board, my_pos, adv_pos, max_step)

            # Find the move with the lowest heuristic
            for move in allMoves:
                # Create a new board
                newBoard = self.setBarrier(chess_board, move)

                # Find the heuristic of the new board
                newHeur = self.minimax(
                    newBoard, my_pos, adv_pos, depth - 1, 1, max_step)

                # If the heuristic is lower than the best heuristic, update the best heuristic
                if newHeur < bestHeur:
                    bestHeur = newHeur

            return bestHeur

        # If it is the opponents turn
        else:
            # Get all the moves they can make
            allMoves = self.getAllMoves(chess_board, adv_pos, my_pos, max_step)

            # Find the move with the highest heuristic
            for move in allMoves:
                # Create a new board
                newBoard = self.setBarrier(chess_board, move)

                # Find the heuristic of the new board
                newHeur = self.minimax(
                    newBoard, my_pos, adv_pos, depth - 1, 0, max_step)

                # If the heuristic is higher than the best heuristic, update the best heuristic
                if newHeur > bestHeur:
                    bestHeur = newHeur

            return bestHeur

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
        minHeur = 10000

        # Create a map of each move to the heuristic
        moveMap = {}

        # Call the minmax function from current board state
        for move in allMoves:
            newBoard = self.setBarrier(chess_board, move)
            currentHeur = self.heuristic(newBoard, adv_pos, move[0])
            # add to moveMap
            moveMap[move] = currentHeur

            if currentHeur < minHeur:
                minHeur = currentHeur
                bestMove = move

        tpi = bestMove

        return bestMove
