# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

# Global Zobrist table for hashing (must be global for consistency)
_ZOBRIST_TABLE = None
_ZOBRIST_SIZE = None

def _init_zobrist(size):
  """Initialize Zobrist hashing table for the given board size."""
  global _ZOBRIST_TABLE, _ZOBRIST_SIZE
  if _ZOBRIST_TABLE is None or _ZOBRIST_SIZE != size:
    _ZOBRIST_SIZE = size
    np.random.seed(42)  # Fixed seed for consistency
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
    # Instance-level TT - fresh for each game, persists across moves within game
    self.tt = {}
    self.tt_access_counter = 0

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
    
    # Initialize Zobrist hashing
    n = chess_board.shape[0]
    _init_zobrist(n)
    
    # Keep most recent TT entries if it gets too large
    if len(self.tt) > 100000:
      # Keep the 50000 most recently accessed entries
      sorted_entries = sorted(self.tt.items(), 
                             key=lambda x: x[1].get('last_access', 0), 
                             reverse=True)
      self.tt = dict(sorted_entries[:50000])
    
    def compute_hash(board):
      """Compute Zobrist hash for the board."""
      h = np.int64(0)
      for piece_val in (1, 2, 3):  # Include obstacles (3)
        coords = np.argwhere(board == piece_val)
        for r, c in coords:
          h ^= _ZOBRIST_TABLE[r, c, piece_val]
      return h

    def evaluate(board, cur_player):
      # Evaluate from cur_player's perspective
      my_count = int((board == cur_player).sum())
      opp_count = int((board == (3 - cur_player)).sum())
      return (my_count - opp_count)

    def is_terminal(board):
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
      
      # 2. Sort remaining moves: duplication first (inline to avoid overhead)
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
      """Compute incremental Zobrist hash for the child position after applying mv.
      Uses Ataxx rules: duplication (dist=1) keeps src; jump (dist=2) empties src.
      Then flip all adjacent opponent discs around destination.
      """
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
              # XOR out opponent, XOR in current player
              h ^= _ZOBRIST_TABLE[rr, cc, opp]
              h ^= _ZOBRIST_TABLE[rr, cc, cur_player]
      return h

    # negamax returns (value, best_move) where value is always from cur_player's perspective
    def negamax(board, cur_player, depth, alpha, beta, board_hash):
      # time check
      if time.time() - start_time > time_limit:
        raise TimeoutError()
      
      # TT lookup
      tt_entry = self.tt.get(board_hash)
      tt_move = None
      if tt_entry and tt_entry['depth'] >= depth:
        # Update access time
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
        # Update access
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
      # If no moves, allow pass: opponent moves next. If both have no moves, terminal will be detected above.
      if not moves:
        # pass: opponent to play at same depth (do not consume depth)
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
        execute_move(nb, mv, cur_player)  # mutates nb
        # Incremental hash for child
        child_h = child_hash(board, cur_player, board_hash, mv)
        val, _ = negamax(nb, 3 - cur_player, depth - 1, -beta, -alpha, child_h)
        val = -val  # Negate because it's from opponent's perspective

        if val > best_value:
          best_value = val
          best_move = mv
        
        alpha = max(alpha, val)
        if alpha >= beta:
          break
      
      # Store in TT with appropriate flag
      if best_value <= original_alpha:
        flag = 'upper'  # All moves were <= alpha (fail-low)
      elif best_value >= beta:
        flag = 'lower'  # We got a cutoff (fail-high)
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
        # time guard before starting a deeper search
        if time.time() - start_time > time_limit:
          break
        val, mv = negamax(chess_board, player, depth, -float('inf'), float('inf'), root_hash)
        # if negamax completed this depth without TimeoutError, accept its move
        if mv is not None:
          last_completed_move = mv
          last_completed_depth = depth
    except TimeoutError:
      # time's up: fall back to last completed depth's move
      pass

    def update_diagnostics(depth):
      """Update and print rolling average depth statistics."""
      stats_file = "depth_stats.txt"
      try:
        with open(stats_file, 'r') as f:
          lines = f.read().strip().split('\n')
          if len(lines) == 2:
            avg_depth = float(lines[0])
            total_moves = int(lines[1])
          else:
            avg_depth = 0.0
            total_moves = 0
      except (FileNotFoundError, ValueError):
        avg_depth = 0.0
        total_moves = 0
      
      # Update rolling average
      total_moves += 1
      avg_depth = ((avg_depth * (total_moves - 1)) + depth) / total_moves
      
      # Write updated stats
      with open(stats_file, 'w') as f:
        f.write(f"{avg_depth}\n{total_moves}\n")
      
      print(f"HeuristicAgent: completed depth {depth}, TT size: {len(self.tt)}, avg depth: {avg_depth:.2f} over {total_moves} moves")
    
    update_diagnostics(last_completed_depth)

    if last_completed_move is None:
      # fallback: pick a random legal move or None if no moves
      return random_move(chess_board, player)
    return last_completed_move