# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys


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
        alreadyVisited = set()

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
            if position in alreadyVisited:
                return
            else:
                alreadyVisited.add(position)

            if depth < max_step:
                # Check if up barrier in the way
                if chess_board[x][y][self.dir_map["u"]] == False:
                    # If not explore that direction
                    dfs((position[0] - 1, position[1]), depth + 1)
                if chess_board[x][y][self.dir_map["d"]] == False:
                    dfs((position[0] + 1, position[1]), depth + 1)
                if chess_board[x][y][self.dir_map["r"]] == False:
                    dfs((position[0], position[1] + 1), depth + 1)
                if chess_board[x][y][self.dir_map["l"]] == False:
                    dfs((position[0], position[1] - 1), depth + 1)

        dfs(my_pos, 0)
        return alreadyVisited

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

        return my_pos, self.dir_map["u"]
