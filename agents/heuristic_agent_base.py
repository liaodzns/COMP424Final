# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("heuristic_agent_base")
class HeuristicAgentBase(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(HeuristicAgentBase, self).__init__()
    self.name = "HeuristicAgentBase"

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

    # Precompute offsets once
    _DUP_OFFSETS = np.array([[dr, dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)], dtype=int)
    _JUMP_OFFSETS = np.array([[dr, dc] for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2) if max(abs(dr), abs(dc)) == 2], dtype=int)
    _OFFSETS = np.vstack((_DUP_OFFSETS, _JUMP_OFFSETS))

    # Simple cache for opponent mobility to avoid recomputation in evaluate
    opp_moves_cache = {}

    def _get_opp_moves_cached(bd):
      # Key by raw bytes; assumes board shape is constant during a game
      key = bd.tobytes()
      cached = opp_moves_cache.get(key)
      if cached is not None:
        return cached
      val = _fast_count_valid_moves(bd, opponent)
      opp_moves_cache[key] = val
      return val

    def _fast_count_valid_moves(board, player):
      # board: numpy array; empty squares == 0
      pieces = np.argwhere(board == player)
      if pieces.size == 0:
        return 0
      n = board.shape[0]

      dests = pieces[:, None, :] + _OFFSETS[None, :, :]  # shape (P, O, 2)
      r = dests[..., 0]; c = dests[..., 1]
      in_bounds = (r >= 0) & (r < n) & (c >= 0) & (c < n)
      if not np.any(in_bounds):
        return 0

      pr = r[in_bounds]; pc = c[in_bounds]
      empties_mask = (board[pr, pc] == 0)
      if not np.any(empties_mask):
        return 0

      return int(np.count_nonzero(empties_mask))

    def evaluate(board):
      # Improved heuristic: material + mobility
      my_count = int((board == player).sum())
      opp_count = int((board == opponent).sum())

      # Opponent Mobility: number of valid moves
      opp_moves = _get_opp_moves_cached(board)

      # Weighted sum
      return (
          1.0 * (my_count - opp_count)
        - 0.5 * (opp_moves)
      )

    def is_terminal(board):
      end, p1, p2 = check_endgame(board)
      return end

    # minimax returns (value, best_move) where value is from root player's perspective
    def minimax(board, cur_player, depth, alpha, beta):
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
        val, _ = minimax(board, 3 - cur_player, depth, alpha, beta)
        return val, None

      best_value = -float('inf') if cur_player == player else float('inf')
      best_move = None

      # Basic move ordering: prefer duplication moves first
      def is_duplication(mv):
        src = mv.get_src(); dst = mv.get_dest()
        return max(abs(dst[0] - src[0]), abs(dst[1] - src[1])) == 1
      moves = sorted(moves, key=lambda mv: (not is_duplication(mv)))

      for mv in moves:
        # time check inside loop
        if time.time() - start_time > time_limit:
          raise TimeoutError()

        nb = board.copy()
        execute_move(nb, mv, cur_player)  # mutates nb
        val, _ = minimax(nb, 3 - cur_player, depth - 1, alpha, beta)

        # val is from root player's perspective
        if cur_player == player:
          # maximize
          if val > best_value:
            best_value = val
            best_move = mv
          alpha = max(alpha, best_value)
          if beta <= alpha:
            break
        else:
          # minimize
          if val < best_value:
            best_value = val
            best_move = mv
          beta = min(beta, best_value)
          if beta <= alpha:
            break

      return best_value, best_move

    last_completed_move = None
    max_depth = 10  # iterative deepening will stop earlier on time
    try:
      for depth in range(1, max_depth + 1):
        # time guard before starting a deeper search
        if time.time() - start_time > time_limit:
          break
        val, mv = minimax(chess_board, player, depth, -float('inf'), float('inf'))
        # if minimax completed this depth without TimeoutError, accept its move
        if mv is not None:
          last_completed_move = mv
          last_completed_depth = depth
    except TimeoutError:
      # time's up: fall back to last completed depth's move
      pass

    print(f"HeuristicAgent: completed depth {last_completed_depth}")

    if last_completed_move is None:
      # fallback: pick a random legal move or None if no moves
      return random_move(chess_board, player)
    return last_completed_move