from contextlib import contextmanager
import logging

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)

def get_adj_positions(board, pos):
    return [(pos[0] + move[0], pos[1] + move[1]) for i, move in enumerate(MOVES) if not board[pos[0], pos[1], i]]

def get_possible_positions_from(chess_board, my_pos, adv_pos, steps, visited=None):
    if visited is None:
        visited = {}
    # if oponents square, not valid
    if my_pos == adv_pos:
        return []
    # check if already visited position with equal or more steps
    if visited.get(my_pos, 0) >= steps:
        return []
    # add self to visited
    visited[my_pos] = steps
    
    # base case if steps is 0 return self moves only
    possible_positions = [my_pos]
    if steps == 0:
        return possible_positions
    
    # recursively visit all posiible adj moves with one less step
    for adj_pos in get_adj_positions(chess_board, my_pos):
        possible_positions = possible_positions + get_possible_positions_from(chess_board, adj_pos, adv_pos, steps-1, visited)
    
    return possible_positions

def get_possible_moves_from(chess_board, my_pos, adv_pos, max_steps):
    return [(ppos, i) for ppos in get_possible_positions_from(chess_board, my_pos, adv_pos, max_steps) for i in range(4) if not chess_board[ppos[0], ppos[1], i]]

def check_endgame(chess_board, my_pos, adv_pos):   
    # check endgame method
    def get_area_from(pos, visited=None):
        if visited is None:
            visited = set()
        if pos in visited:
            return set()
        visited.add(pos)
        for new_pos in get_adj_positions(chess_board, pos):
            visited = visited | get_area_from(new_pos, visited)
        return visited
    
    my_area = get_area_from(my_pos)
    if adv_pos in my_area:
        return False, len(my_area), len(my_area)
    return True, len(my_area), len(get_area_from(adv_pos))