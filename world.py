import numpy as np
from copy import deepcopy
import traceback
from agents import *
from ui import UIEngine
from time import sleep, time
import click
import logging
from store import AGENT_REGISTRY
from constants import *
import sys

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


class World:
    def __init__(
        self,
        player_1="random_agent",
        player_2="random_agent",
        board_size=None,
        display_ui=False,
        display_delay=2,
        display_save=False,
        display_save_path=None,
        autoplay=False,
    ):
        """
        Initialize the game world

        Parameters
        ----------
        player_1: str
            The registered class of the first player
        player_2: str
            The registered class of the second player
        board_size: int
            The size of the board. If None, board_size = a number between MIN_BOARD_SIZE and MAX_BOARD_SIZE
        display_ui : bool
            Whether to display the game board
        display_delay : float
            Delay between each step
        display_save : bool
            Whether to save an image of the game board
        display_save_path : str
            The path to save the image
        autoplay : bool
            Whether the game is played in autoplay mode
        """
        # Two players
        logger.info("Initialize the game world")
        # Load agents as defined in decorators
        self.player_1_name = player_1
        self.player_2_name = player_2
        if player_1 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_1}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )
        if player_2 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_2}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )

        p0_agent = AGENT_REGISTRY[player_1]
        p1_agent = AGENT_REGISTRY[player_2]
        logger.info(f"Registering p0 agent : {player_1}")
        self.p0 = p0_agent()
        logger.info(f"Registering p1 agent : {player_2}")
        self.p1 = p1_agent()

        # check autoplay
        if autoplay:
            if not self.p0.autoplay or not self.p1.autoplay:
                raise ValueError(
                    f"Autoplay mode is not supported by one of the agents ({self.p0} -> {self.p0.autoplay}, {self.p1} -> {self.p1.autoplay}). Please set autoplay=True in the agent class."
                )

        self.player_names = {PLAYER_1_ID: PLAYER_1_NAME, PLAYER_2_ID: PLAYER_2_NAME}
        self.dir_names = {
            DIRECTION_UP: DIRECTION_UP_NAME,
            DIRECTION_RIGHT: DIRECTION_RIGHT_NAME,
            DIRECTION_DOWN: DIRECTION_DOWN_NAME,
            DIRECTION_LEFT: DIRECTION_LEFT_NAME,
        }

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        if board_size is None:
            # Random chessboard size
            self.board_size = np.random.randint(MIN_BOARD_SIZE, MAX_BOARD_SIZE)
            logger.info(
                f"No board size specified. Randomly generating size : {self.board_size}x{self.board_size}"
            )
        else:
            self.board_size = board_size
            logger.info(f"Setting board size to {self.board_size}x{self.board_size}")

        # Index in dim2 represents [Up, Right, Down, Left] respectively
        # Record barriers and boarders for each block
        self.chess_board = np.zeros((self.board_size, self.board_size, 4), dtype=bool)

        # Set borders
        self.chess_board[0, :, 0] = True
        self.chess_board[:, 0, 3] = True
        self.chess_board[-1, :, 2] = True
        self.chess_board[:, -1, 1] = True

        # Maximum Steps
        self.max_step = (self.board_size + 1) // 2

        # Random barriers (symmetric)
        for _ in range(self.max_step):
            pos = np.random.randint(0, self.board_size, size=2)
            r, c = pos
            dir = np.random.randint(0, 4)
            while self.chess_board[r, c, dir]:
                pos = np.random.randint(0, self.board_size, size=2)
                r, c = pos
                dir = np.random.randint(0, 4)
            anti_pos = self.board_size - 1 - pos
            anti_dir = self.opposites[dir]
            anti_r, anti_c = anti_pos
            self.set_barrier(r, c, dir)
            self.set_barrier(anti_r, anti_c, anti_dir)

        # Random start position (symmetric but not overlap)
        self.p0_pos = np.random.randint(0, self.board_size, size=2)
        self.p1_pos = self.board_size - 1 - self.p0_pos
        while np.array_equal(self.p0_pos, self.p1_pos):
            self.p0_pos = np.random.randint(0, self.board_size, size=2)
            self.p1_pos = self.board_size - 1 - self.p0_pos
        # Whose turn to step
        self.turn = 0

        # Check initialization
        self.initial_end, _, _ = self.check_endgame()

        # Time taken by each player
        self.p0_time = 0
        self.p1_time = 0

        # Cache to store and use the data
        self.results_cache = ()
        # UI Engine
        self.display_ui = display_ui
        self.display_delay = display_delay
        self.display_save = display_save
        self.display_save_path = display_save_path
        if display_ui:
            # Initialize UI Engine
            logger.info(
                f"Initializing the UI Engine, with display_delay={display_delay} seconds"
            )
            self.ui_engine = UIEngine(self.board_size, self)
            self.render()

    def get_current_player(self):
        """
        Get the positions of the current player

        Returns
        -------
        tuple of (current_player_obj, current_player_pos, adversary_player_pos)
        """
        if not self.turn:
            return self.p0, self.p0_pos, self.p1_pos
        else:
            return self.p1, self.p1_pos, self.p0_pos

    def update_player_time(self, time_taken):
        """
        Update the time taken by the player

        Parameters
        ----------
        time_taken : float
            Time taken by the player
        """
        if not self.turn:
            self.p0_time += time_taken
        else:
            self.p1_time += time_taken

    def step(self):
        """
        Take a step in the game world.
        Runs the agents' step function and update the game board accordingly.
        If the agents' step function raises an exception, the step will be replaced by a Random Walk.

        Returns
        -------
        results: tuple
            The results of the step containing (is_endgame, player_1_score, player_2_score)
        """
        cur_player, cur_pos, adv_pos = self.get_current_player()

        try:
            # Run the agents step function
            start_time = time()
            next_pos, dir = cur_player.step(
                deepcopy(self.chess_board),
                tuple(cur_pos),
                tuple(adv_pos),
                self.max_step,
            )
            self.update_player_time(time() - start_time)

            next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)
            if not self.check_boundary(next_pos):
                raise ValueError("End position {} is out of boundary".format(next_pos))
            if not 0 <= dir <= 3:
                raise ValueError(
                    "Barrier dir should reside in [0, 3], but your dir is {}".format(
                        dir
                    )
                )
            if not self.check_valid_step(cur_pos, next_pos, dir):
                raise ValueError(
                    "Not a valid step from {} to {} and put barrier at {}, with max steps = {}".format(
                        cur_pos, next_pos, dir, self.max_step
                    )
                )
        except BaseException as e:
            ex_type = type(e).__name__
            if (
                "SystemExit" in ex_type and isinstance(cur_player, HumanAgent)
            ) or "KeyboardInterrupt" in ex_type:
                sys.exit(0)
            print(
                "An exception raised. The traceback is as follows:\n{}".format(
                    traceback.format_exc()
                )
            )
            print("Execute Random Walk!")
            next_pos, dir = self.random_walk(tuple(cur_pos), tuple(adv_pos))
            next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        # Print out each step
        # print(self.turn, next_pos, dir)
        logger.info(
            f"Player {self.player_names[self.turn]} moves to {next_pos} facing {self.dir_names[dir]}"
        )
        if not self.turn:
            self.p0_pos = next_pos
        else:
            self.p1_pos = next_pos
        # Set the barrier to True
        r, c = next_pos
        self.set_barrier(r, c, dir)

        # Change turn
        self.turn = 1 - self.turn

        results = self.check_endgame()
        self.results_cache = results

        # Print out Chessboard for visualization
        if self.display_ui:
            self.render()
            if results[0]:
                # If game ends and displaying the ui, wait for user input
                click.echo("Press a button to exit the game.")
                try:
                    _ = click.getchar()
                except:
                    _ = input()
        return results

    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : np.ndarray
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
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
        if player_win >= 0:
            logging.info(
                f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            )
        else:
            logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score

    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def random_walk(self, my_pos, adv_pos):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, self.max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def render(self, debug=False):
        """
        Render the game board using the UI Engine
        """
        self.ui_engine.render(self.chess_board, self.p0_pos, self.p1_pos, debug=debug)
        sleep(self.display_delay)


if __name__ == "__main__":
    world = World()
    is_end, p0_score, p1_score = world.step()
    while not is_end:
        is_end, p0_score, p1_score = world.step()
    print(p0_score, p1_score)
