# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("heuristic_agent_basic")
class HeuristicAgentBasic(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(HeuristicAgentBasic, self).__init__()
    self.name = "HeuristicAgentBasic"

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

    # Basic heuristic to start:
    # (my_disc_count - opponent_disc_count) after the move.
    start_time = time.time()

    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
      return random_move(chess_board, player)

    best_move = None
    best_score = -float('inf')

    for move in valid_moves:
      # time guard: don't exceed 1.999s
      if time.time() - start_time > 1.999:
        break
      sim_board = deepcopy(chess_board)
      execute_move(sim_board, move, player)
      my_count = int((sim_board == player).sum())
      opp_count = int((sim_board == opponent).sum())
      score = my_count - opp_count

      score = float(score) + float(np.random.uniform(-1e-6, 1e-6))

      if score > best_score:
        best_score = score
        best_move = move

    time_taken = time.time() - start_time
    print("HeuristicAgent decision time:", round(time_taken, 4), "s")

    if best_move is None:
      return random_move(chess_board, player)
    return best_move