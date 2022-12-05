# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent

MAX_DEPTH = 2


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.max_depth=MAX_DEPTH
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        best_move, rating = self.min_max(chess_board, my_pos, adv_pos, max_step+1, 0) # I could not tell you why its max step + 1 but here we are
        # print(f"move rating : {rating}")
        return best_move

    def rate_move(self, board, my_pos, adv_pos, max_step, move, turn, alpha, beta, current_depth=0):
        # apply move to board
        (x, y), d = move
        board[x, y, d] = True
        board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = True
        
        # check endgame
        is_game_over, my_score, adv_score = self.check_endgame(board, my_pos, adv_pos)
        
        # if position results in game ending, return score
        if is_game_over:
            # return board to original position
            board[x, y, d] = False
            board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = False
            
            if my_score > adv_score:
                return 100
            elif my_score < adv_score:
                return -100
            else:
                return 0
        
        # if out of depth
        if current_depth >= self.max_depth:
            score = self.naive_board_estimator(board, my_pos, adv_pos, max_step)
            board[x, y, d] = False
            board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = False
            return score
        
        # get rating for move
        if turn==0:
            my_pos = (x,y)
            _, rating=self.min_max(board, my_pos, adv_pos, max_step, 1, alpha=alpha, beta=beta, current_depth=current_depth)
        
        elif turn==1:
            adv_pos = (x,y)
            _, rating=self.min_max(board, my_pos, adv_pos, max_step, 0, alpha=alpha, beta=beta, current_depth=current_depth)

        board[x, y, d] = False
        board[x + self.moves[d][0], y + self.moves[d][1], (d+2)%4] = False

        # return rating
        return rating

    def min_max(self, board, my_pos, adv_pos, max_step, turn, current_depth=0, alpha=-999, beta=999):
        # player 1s turn, get max
        if turn==0:
            possible_moves = self.get_possible_moves_from(board, my_pos, adv_pos, max_step)
            best_move = possible_moves[0]
            for move in possible_moves:
                move_score = self.rate_move(board, my_pos, adv_pos, max_step, move, turn, alpha, beta, current_depth=current_depth+1)
                if move_score > alpha:
                    best_move = move
                    alpha = move_score
                if alpha >= beta:
                    return move, beta
            return best_move, alpha

        # player 2's turn, get minimum
        elif turn==1:
            possible_moves = self.get_possible_moves_from(board, adv_pos, my_pos, max_step)
            best_move = possible_moves[0]
            for move in possible_moves:
                move_score = self.rate_move(board, my_pos, adv_pos, max_step, move, turn, alpha, beta, current_depth=current_depth+1)
                if move_score < beta:
                    best_move = move
                    beta = move_score
                if alpha >= beta:
                    return move, alpha
            return best_move, beta

    def naive_board_estimator(self, board, my_pos, adv_pos, max_step):
        # naive board value estimation, based on number of possible positions
        return len(self.get_possible_positions_from(board, my_pos, adv_pos, max_step))-len(self.get_possible_positions_from(board, adv_pos, my_pos, max_step))

    def get_possible_positions_from(self, chess_board, pos, adv_pos, steps, visited=None):
        if visited is None:
            visited = {}
        # if oponents square, not valid
        if pos == adv_pos:
            return []
        # check if already visited position with equal or more steps
        if visited.get(pos, 0) >= steps:
            return []
        # add self to visited
        visited[pos] = steps
        
        # base case if steps is 0 return self moves only
        possible_positions = [pos]
        if steps == 0:
            return possible_positions
        
        # recursively visit all posiible adj moves with one less step
        for adj_pos in self.get_adj_positions(chess_board, pos):
            possible_positions = possible_positions + self.get_possible_positions_from(chess_board, adj_pos, adv_pos, steps-1, visited)
        
        return possible_positions

    def get_possible_moves_from(self, chess_board, pos, adv_pos, max_steps):
        return [(ppos, i) for ppos in self.get_possible_positions_from(chess_board, pos, adv_pos, max_steps) for i in range(4) if not chess_board[ppos[0], ppos[1], i]]

    def check_endgame(self, chess_board, my_pos, adv_pos):   
        # check board to identify if the game is ended
        def get_area_from(pos, visited=None):
            if visited is None:
                visited = set()
            if pos in visited:
                return set()
            visited.add(pos)
            for new_pos in self.get_adj_positions(chess_board, pos):
                visited = visited | get_area_from(new_pos, visited)
            return visited
        
        my_area = get_area_from(my_pos)
        if adv_pos in my_area:
            return False, len(my_area), len(my_area)
        return True, len(my_area), len(get_area_from(adv_pos))

    def get_adj_positions(self, board, pos):
        return [(pos[0] + move[0], pos[1] + move[1]) for i, move in enumerate(self.moves) if not board[pos[0], pos[1], i]]

    def in_board(self, chess_board, pos):
        return 0 <= pos[0] < chess_board.shape[0] and 0 <= pos[1] < chess_board.shape[1]
