# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

# Global Zobrist table for hashing
_ZOBRIST_TABLE = None
_ZOBRIST_SIZE = None

def _init_zobrist(size):
  """Initialize Zobrist hashing table for the given board size."""
  global _ZOBRIST_TABLE, _ZOBRIST_SIZE
  if _ZOBRIST_TABLE is None or _ZOBRIST_SIZE != size:
    _ZOBRIST_SIZE = size
    np.random.seed(42)
    # Table for each position and each piece type (0=empty, 1=player1, 2=player2, 3=obstacle)
    _ZOBRIST_TABLE = np.random.randint(0, 2**63, size=(size, size, 4), dtype=np.int64)

@register_agent("heuristic_agent")
class HeuristicAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(HeuristicAgent, self).__init__()
    self.name = "HeuristicAgent"
    # Fresh TT for each game, persists across moves within games
    self.tt = {}
    self.tt_access_counter = 0

  def step(self, chess_board, player, opponent):
    """ 
    Iterative-deepening minimax. Using material-only heuristic at leaf 
    nodes. Stop when elapsed time > 1.9s and return the best move found
    at the last fully completed depth.
    """
    start_time = time.time()
    time_limit = 1.9
    
    # Initialize Zobrist hashing
    n = chess_board.shape[0]
    _init_zobrist(n)
    
    # Keep most recent TT entries if it gets too large
    if len(self.tt) > 100000:
      # Keep the 50000 most recently accessed entries
      sorted_entries = sorted(self.tt.items(), key=lambda x: x[1].get('last_access', 0), reverse=True)
      self.tt = dict(sorted_entries[:50000])
    
    def compute_hash(board):
      """Compute Zobrist hash for the board."""
      h = np.int64(0)
      for piece_val in (1, 2, 3):
        coords = np.argwhere(board == piece_val)
        for r, c in coords:
          h ^= _ZOBRIST_TABLE[r, c, piece_val]
      return h

    def evaluate(board, cur_player):
      """Simple material count heuristic."""
      my_count = int((board == cur_player).sum())
      opp_count = int((board == (3 - cur_player)).sum())
      return (my_count - opp_count)

    def is_terminal(board):
      """Check if the game has ended."""
      end, p1, p2 = check_endgame(board)
      return end
    
    def order_moves(moves, tt_move):
      """Order moves: TT move first, then duplications, then jumps."""
      ordered_moves = []
      
      # 1. Try TT move first if it exists in current move list
      if tt_move:
        for mv in moves:
          try:
            if (mv.get_src() == tt_move.get_src() and 
                mv.get_dest() == tt_move.get_dest()):
              ordered_moves.append(mv)
              break
          except:
            pass
      
      # 2. Sort remaining moves: duplication first
      remaining = [mv for mv in moves if mv not in ordered_moves]
      dup_moves = []
      jump_moves = []
      for mv in remaining:
        try:
          src = mv.get_src()
          dst = mv.get_dest()
          dist = max(abs(dst[0] - src[0]), abs(dst[1] - src[1]))
          if dist == 1:
            dup_moves.append(mv)
          else:
            jump_moves.append(mv)
        except:
          jump_moves.append(mv)
      
      ordered_moves.extend(dup_moves)
      ordered_moves.extend(jump_moves)
      return ordered_moves

    def child_hash(parent_board, cur_player, parent_hash, mv):
      """Compute incremental Zobrist hash for the child position after applying move."""
      h = np.int64(parent_hash)
      src_r, src_c = mv.get_src()
      dst_r, dst_c = mv.get_dest()
      opp = 3 - cur_player

      # Determine duplication vs jump by Chebyshev distance
      dist = max(abs(dst_r - src_r), abs(dst_c - src_c))

      # For jump, remove piece from source
      if dist > 1:
        h ^= _ZOBRIST_TABLE[src_r, src_c, cur_player]

      # Place piece at destination
      h ^= _ZOBRIST_TABLE[dst_r, dst_c, cur_player]

      # Flip adjacent opponent pieces around destination
      nrows, ncols = parent_board.shape
      for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
          if dr == 0 and dc == 0:
            continue
          rr = dst_r + dr
          cc = dst_c + dc
          if 0 <= rr < nrows and 0 <= cc < ncols:
            if parent_board[rr, cc] == opp:
              h ^= _ZOBRIST_TABLE[rr, cc, opp]
              h ^= _ZOBRIST_TABLE[rr, cc, cur_player]
      return h

    def negamax(board, cur_player, depth, alpha, beta, board_hash):
      """Negamax with alpha-beta pruning and transposition table."""
      # time check
      if time.time() - start_time > time_limit:
        raise TimeoutError()
      
      # TT lookup
      tt_entry = self.tt.get(board_hash)
      tt_move = None
      if tt_entry and tt_entry['depth'] >= depth:
        self.tt_access_counter += 1
        tt_entry['last_access'] = self.tt_access_counter
        
        flag, value = tt_entry['flag'], tt_entry['value']
        if flag == 'exact':
          return value, tt_entry.get('best_move')
        elif flag == 'lower' and value >= beta:
          return value, tt_entry.get('best_move')
        elif flag == 'upper' and value <= alpha:
          return value, tt_entry.get('best_move')
      
      # Remember TT move for move ordering
      if tt_entry:
        self.tt_access_counter += 1
        tt_entry['last_access'] = self.tt_access_counter
        tt_move = tt_entry.get('best_move')

      # terminal or depth limit
      if depth == 0 or is_terminal(board):
        val = evaluate(board, cur_player)
        self.tt_access_counter += 1
        self.tt[board_hash] = {
          'depth': depth, 
          'value': val, 
          'flag': 'exact', 
          'best_move': None,
          'last_access': self.tt_access_counter
        }
        return val, None

      moves = get_valid_moves(board, cur_player)
      # If no moves, allow pass: opponent moves next
      if not moves:
        val, _ = negamax(board, 3 - cur_player, depth, -beta, -alpha, board_hash)
        return -val, None

      best_value = -float('inf')
      best_move = None
      original_alpha = alpha

      # Move ordering
      ordered_moves = order_moves(moves, tt_move)

      for mv in ordered_moves:
        # time check inside loop
        if time.time() - start_time > time_limit:
          raise TimeoutError()

        nb = board.copy()
        execute_move(nb, mv, cur_player)
        # Incremental hash for child
        child_h = child_hash(board, cur_player, board_hash, mv)
        val, _ = negamax(nb, 3 - cur_player, depth - 1, -beta, -alpha, child_h)
        val = -val

        if val > best_value:
          best_value = val
          best_move = mv
        
        alpha = max(alpha, val)
        if alpha >= beta:
          break
      
      # Store in TT with appropriate flag
      if best_value <= original_alpha:
        flag = 'upper'  # All moves were <= alpha
      elif best_value >= beta:
        flag = 'lower'  # We got a cutoff
      else:
        flag = 'exact'  # Value is within window
      
      self.tt_access_counter += 1
      self.tt[board_hash] = {
        'depth': depth,
        'value': best_value,
        'flag': flag,
        'best_move': best_move,
        'last_access': self.tt_access_counter
      }

      return best_value, best_move

    last_completed_move = None
    last_completed_depth = 0
    max_depth = 8  # iterative deepening will stop earlier on time
    try:
      # Compute root hash once
      root_hash = compute_hash(chess_board)
      for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
          break
        val, mv = negamax(chess_board, player, depth, -float('inf'), float('inf'), root_hash)
        # if negamax completed this depth without TimeoutError, accept its move
        if mv is not None:
          last_completed_move = mv
          last_completed_depth = depth
    except TimeoutError:
      pass

    if last_completed_move is None:
      # fallback: pick a random legal move or None if no moves
      return random_move(chess_board, player)
    return last_completed_move