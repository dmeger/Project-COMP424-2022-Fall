# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import math
import copy


@register_agent("sample_agent")
class SampleAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(SampleAgent, self).__init__()
        self.name = "SampleAgent"
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

    # Copied and adjusted code from world.py
    # Prof said we could do this https://edstem.org/us/courses/28046/discussion/2187454
    def checkEndGame(self, chess_board, my_pos, adv_pos):
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
        p0_pos, wall = my_pos
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

    def heuristic(self, chess_board, adv_pos, new_move):
        # Create a copy of the bo
        boardAfterMove = self.setBarrier(chess_board, new_move)

        # Check if someone can win the game
        # This function is interpreted from world.py
        someoneWon, myScore, theirScore = self.checkEndGame(
            boardAfterMove, new_move, adv_pos)
        if myScore > theirScore:
            return -1000 - myScore
        if myScore < theirScore:
            return 1000 + myScore

        # Initialize a heuris
        heuristic = math.dist(new_move[0], adv_pos)

        # Count the # of walls that surround the position
        wallCount = 0
        new_pos, dir = new_move
        x, y = new_pos
        for wall in boardAfterMove[x][y]:
            if wall == True:
                wallCount = wallCount + 1

        heuristic = heuristic + (wallCount * 10)

        return heuristic

    # function for finding the best move based on the heuristic
    def findBestMove(self, allMoves, chess_board, my_pos, adv_pos, max_step):
        # initialize the best move as the first move
        min_H_move = allMoves.pop()
        # calculate the heuristic value
        min_H = self.heuristic(chess_board, adv_pos, min_H_move)
        for new_move in allMoves:
            # go through all the possible values
            new_H = self.heuristic(chess_board, adv_pos, new_move)
            # if you find a move with a better heuristic set it as best move
            if (new_H < min_H):
                min_H = new_H
                min_H_move = new_move
        # return best possible move
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
