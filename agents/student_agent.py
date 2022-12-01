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

    def is_valid_move(self, chess_board, start_pos, end_pos, board_dir, adv_pos, max_step):

        r, c = end_pos
        # print("r " + str(r))
        # print("c " + str(c))
        if r < 0 or r >= len(chess_board) or c < 0 or c >= len(chess_board):
            return False

        if chess_board[r, c, board_dir] == True:
            return False

        if end_pos == adv_pos:
            return False

        hor_dist = abs(end_pos[0] - start_pos[0])
        ver_dist = abs(end_pos[1] - start_pos[1])
        moves = hor_dist + ver_dist

        if moves > max_step:
            return False

        return True

    def get_legal_moves(self, chess_board, my_pos, adv_pos, max_step):
        legal = []
        for i in range(my_pos[0], my_pos[0] + max_step):
            for j in range(my_pos[1], my_pos[0] + max_step):
                for k in range(0, 4):
                    if self.is_valid_move(chess_board, my_pos, (i, j), k, adv_pos, max_step):
                        legal.append(((i, j), k))
        return legal


    def eval_move(self, chess_board, start_pos, end_pos, board_dir, adv_pos):

        points = 0

        start_to_adv_x = abs(adv_pos[0] - start_pos[0])
        start_to_adv_y = abs(adv_pos[1] - start_pos[1])
        start_to_adv_dist = start_to_adv_x + start_to_adv_y

        end_to_adv_x = abs(adv_pos[0] - end_pos[0])
        end_to_adv_y = abs(adv_pos[1] - end_pos[1])
        end_to_adv_dist = end_to_adv_x + end_to_adv_y

        if end_to_adv_dist < start_to_adv_dist:
            points += 10
        else:
            points -= 10

        adv_x_rotation = adv_pos[0] - end_pos[0]
        adv_y_rotation = adv_pos[1] - end_pos[1]

        if adv_x_rotation < 0:
            if board_dir == 0:
                points += 5
            else:
                points -= 5

        if adv_x_rotation > 0:
            if board_dir == 2:
                points += 5
            else:
                points -= 5

        if adv_y_rotation > 0:
            if board_dir == 1:
                points += 5
            else:
                points -= 5
        if adv_y_rotation < 0:
            if board_dir == 3:
                points += 5
            else: points -= 5

        return chess_board, points

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
        return my_pos, self.dir_map["u"]
