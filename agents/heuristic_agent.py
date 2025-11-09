# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("heuristic_agent")
class HeuristicAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(HeuristicAgent, self).__init__()
    self.name = "HeuristicAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Iterative-deepening minimax. Using material-only heuristic
    # at leaf nodes. Stop when elapsed time > 1.9s and return the best move
    # found at the last fully completed depth.
    start_time = time.time()
    time_limit = 1.9

    def evaluate(board):
      # material-only heuristic from the root player's perspective
      my_count = int((board == player).sum())
      opp_count = int((board == opponent).sum())
      return float(my_count - opp_count)

    def is_terminal(board):
      end, p1, p2 = check_endgame(board)
      return end

    # minimax returns (value, best_move) where value is from root player's perspective
    def minimax(board, cur_player, depth):
      # time check
      if time.time() - start_time > time_limit:
        raise TimeoutError()

      # terminal or depth limit
      if depth == 0 or is_terminal(board):
        return evaluate(board), None

      moves = get_valid_moves(board, cur_player)
      # If no moves, allow pass: opponent moves next. If both have no moves, terminal will be detected above.
      if not moves:
        # pass: opponent to play at same depth (do not consume depth)
        val, _ = minimax(board, 3 - cur_player, depth)
        return val, None

      best_value = -float('inf') if cur_player == player else float('inf')
      best_move = None

      for mv in moves:
        # time check inside loop
        if time.time() - start_time > time_limit:
          raise TimeoutError()

        nb = deepcopy(board)
        execute_move(nb, mv, cur_player)  # mutates nb
        val, _ = minimax(nb, 3 - cur_player, depth - 1)

        # val is from root player's perspective
        if cur_player == player:
          # maximize
          if val > best_value:
            best_value = val
            best_move = mv
        else:
          # minimize
          if val < best_value:
            best_value = val
            best_move = mv

      return best_value, best_move

    # iterative deepening
    last_completed_move = None
    max_depth = 6  # reasonable cap; iterative deepening will stop earlier on time
    try:
      for depth in range(1, max_depth + 1):
        # time guard before starting a deeper search
        if time.time() - start_time > time_limit:
          break
        val, mv = minimax(chess_board, player, depth)
        # if minimax completed this depth without TimeoutError, accept its move
        if mv is not None:
          last_completed_move = mv
    except TimeoutError:
      # time's up: fall back to last completed depth's move
      pass

    if last_completed_move is None:
      # fallback: pick a random legal move or None if no moves
      return random_move(chess_board, player)
    return last_completed_move