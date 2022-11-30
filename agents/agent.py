class Agent:
    def __init__(self):
        """
        Initialize the agent, add a name which is used to register the agent
        """
        self.name = "DummyAgent"
        # Flag to indicate whether the agent can be used to autoplay
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Main decision logic of the agent, which is called by the simulator.
        Extend this method to implement your own agent to play the game.

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size, 4)
            The chess board.
        my_pos : tuple of int
            The position of the agent.
        adv_pos : tuple of int
            The position of the adversary (opponent).
        max_step : int
            The maximum number of steps that the agent can take.

        Returns
        -------
        my_pos : tuple of int
            The new position of the agent.
        dir : int
            The direction of the agent, as defined in world.py (DIRECTION_UP/DIRECTION_DOWN/DIRECTION_LEFT/DIRECTION_RIGHT).
        """
        pass
