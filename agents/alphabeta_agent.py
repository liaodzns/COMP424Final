# Alpha-Beta Pruning Agent for Ataxx Game
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("alphabeta_agent")
class AlphabetaAgent(Agent):
  """
  Agent that uses Minimax algorithm with alpha-beta pruning
  to select optimal moves, along with iterative deepening and needed heuristics.
  """

  def __init__(self):
    super(AlphabetaAgent, self).__init__()
    self.name = "AlphabetaAgent"
    self.autoplay = True
    self.time_limit = 1.95  

  def step(self, chess_board, player, opponent):
    """
    Main decision function that selects the best move using Minimax with alpha-beta pruning.
    
    Parameters:
    - chess_board: numpy array representing the game board (0=empty, 1=player1, 2=player2, 3=obstacle)
    - player: integer representing this agent's player number (1 or 2)
    - opponent: integer representing the opponent's player number (1 or 2)
    
    Returns:
    - MoveCoordinates: the selected move (source and destination positions)
    """
    
    # Record the start time to ensure we don't exceed the time limit
    start_time = time.time()
    
    # Get all valid moves available to the current player
    valid_moves = get_valid_moves(chess_board, player)
    
    # If no valid moves exist, return None (this should be rare in Ataxx)
    if not valid_moves:
      return None
    
    # If only one move is available, return it immediately without searching
    if len(valid_moves) == 1:
      return valid_moves[0]
    
    # Initialize the best move to the first valid move as a fallback
    best_move = valid_moves[0]
    # Initialize best score to negative infinity (we're maximizing)
    best_score = float('-inf')
    
    # Use iterative deepening: start with shallow depth and increase gradually
    # This ensures we always have a move even if time runs out during a deeper search
    max_depth = 1
    
    # Continue searching deeper until we run out of time
    while True:
      # Check if we have enough time for another iteration (need at least 0.1 seconds)
      time_elapsed = time.time() - start_time
      if time_elapsed > self.time_limit - 0.1:
        break
      
      # Store the current best move in case this depth iteration doesn't complete
      iteration_best_move = best_move
      iteration_best_score = float('-inf')
      
      # Evaluate each valid move at the current depth
      for move in valid_moves:
        # Check if we're running out of time
        if time.time() - start_time > self.time_limit - 0.05:
          break
        
        # Create a deep copy of the board to simulate the move
        simulated_board = deepcopy(chess_board)
        
        # Execute the move on the simulated board
        execute_move(simulated_board, move, player)
        
        # Run minimax with alpha-beta pruning starting from opponent's turn (minimizing)
        # Initial alpha = -infinity, beta = +infinity
        score = self.minimax(simulated_board, max_depth - 1, float('-inf'), float('inf'), 
                            False, player, opponent, start_time)
        
        # If this move has a better score, update our best move
        if score > iteration_best_score:
          iteration_best_score = score
          iteration_best_move = move
      
      # Update the best move and score if this iteration completed
      if iteration_best_score > best_score:
        best_score = iteration_best_score
        best_move = iteration_best_move
      
      # Increase depth for next iteration
      max_depth += 1
      
      # Stop if we've reached a very deep search (unlikely to reach this)
      if max_depth > 50:
        break
    
    # Calculate and print timing information
    time_taken = time.time() - start_time
    print(f"AlphabetaAgent's turn took {time_taken:.3f} seconds, searched to depth {max_depth-1}")
    
    # Return the best move found
    return best_move

  def minimax(self, chess_board, depth, alpha, beta, is_maximizing, player, opponent, start_time):
    """
    Minimax algorithm with alpha-beta pruning to find the optimal move value.
    
    Parameters:
    - chess_board: current board state
    - depth: remaining depth to search
    - alpha: best score for maximizing player (lower bound for maximizer)
    - beta: best score for minimizing player (upper bound for minimizer)
    - is_maximizing: True if current player is maximizing, False if minimizing
    - player: the agent's player number (the maximizing player)
    - opponent: the opponent's player number (the minimizing player)
    - start_time: start time of the search (for time management)
    
    Returns:
    - float: the evaluation score of the position
    """
    
    # Check if we're running out of time; if so, return heuristic evaluation
    if time.time() - start_time > self.time_limit - 0.05:
      return self.evaluate_board(chess_board, player, opponent)
    
    # Check if the game has ended
    is_endgame, player_score, opponent_score = check_endgame(chess_board)
    
    # If game is over, return a very large score based on who won
    if is_endgame:
      if player_score > opponent_score:
        # We won - return large positive score, bonus for winning earlier
        return 100000 + depth * 1000
      elif player_score < opponent_score:
        # We lost - return large negative score, penalty for losing later
        return -100000 - depth * 1000
      else:
        # Draw - return neutral score
        return 0
    
    # If we've reached the depth limit, return heuristic evaluation
    if depth == 0:
      return self.evaluate_board(chess_board, player, opponent)
    
    # Determine which player's turn it is
    current_player = player if is_maximizing else opponent
    
    # Get all valid moves for the current player
    valid_moves = get_valid_moves(chess_board, current_player)
    
    # If no valid moves, the player passes (should be rare in Ataxx)
    if not valid_moves:
      # Recursively evaluate with the other player's turn
      return self.minimax(chess_board, depth - 1, alpha, beta, not is_maximizing, 
                         player, opponent, start_time)
    
    # Maximizing player's turn (our agent trying to maximize score)
    if is_maximizing:
      # Initialize max_eval to negative infinity
      max_eval = float('-inf')
      
      # Try each valid move
      for move in valid_moves:
        # Create a copy of the board to simulate the move
        simulated_board = deepcopy(chess_board)
        # Execute the move
        execute_move(simulated_board, move, current_player)
        
        # Recursively evaluate the resulting position (opponent's turn, minimizing)
        eval_score = self.minimax(simulated_board, depth - 1, alpha, beta, False, 
                                  player, opponent, start_time)
        
        # Update max_eval if this move is better
        max_eval = max(max_eval, eval_score)
        
        # Update alpha (best score for maximizer so far)
        alpha = max(alpha, eval_score)
        
        # Alpha-beta pruning: if beta <= alpha, the minimizer will never choose this branch
        if beta <= alpha:
          break  # Prune remaining moves
      
      return max_eval
    
    # Minimizing player's turn (opponent trying to minimize our score)
    else:
      # Initialize min_eval to positive infinity
      min_eval = float('inf')
      
      # Try each valid move
      for move in valid_moves:
        # Create a copy of the board to simulate the move
        simulated_board = deepcopy(chess_board)
        # Execute the move
        execute_move(simulated_board, move, current_player)
        
        # Recursively evaluate the resulting position (our turn, maximizing)
        eval_score = self.minimax(simulated_board, depth - 1, alpha, beta, True, 
                                  player, opponent, start_time)
        
        # Update min_eval if this move is better for minimizer
        min_eval = min(min_eval, eval_score)
        
        # Update beta (best score for minimizer so far)
        beta = min(beta, eval_score)
        
        # Alpha-beta pruning: if beta <= alpha, the maximizer will never choose this branch
        if beta <= alpha:
          break  # Prune remaining moves
      
      return min_eval

  def evaluate_board(self, chess_board, player, opponent):
    """
    Heuristic evaluation function to score a board position.
    Higher scores are better for the player.
    
    Parameters:
    - chess_board: the current board state
    - player: the agent's player number
    - opponent: the opponent's player number
    
    Returns:
    - float: the heuristic score of the position
    """
    
    # Count pieces for each player
    player_pieces = np.count_nonzero(chess_board == player)
    opponent_pieces = np.count_nonzero(chess_board == opponent)
    
    # If opponent has no pieces, we win - return very high score
    if opponent_pieces == 0:
      return 100000
    # If we have no pieces, we lose - return very low score
    if player_pieces == 0:
      return -100000
    
    # Calculate piece difference (material advantage)
    piece_difference = player_pieces - opponent_pieces
    
    # Calculate mobility (number of valid moves available)
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility_difference = player_moves - opponent_moves
    
    # Get board dimensions
    board_size = chess_board.shape[0]
    
    # Evaluate corner control (corners are strategic in Ataxx)
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
    player_corners = sum(1 for (r, c) in corners if chess_board[r, c] == player)
    opponent_corners = sum(1 for (r, c) in corners if chess_board[r, c] == opponent)
    corner_advantage = player_corners - opponent_corners
    
    # Evaluate edge control (edges are also valuable)
    player_edges = 0
    opponent_edges = 0
    # Count pieces on top and bottom edges
    for c in range(board_size):
      if chess_board[0, c] == player:
        player_edges += 1
      elif chess_board[0, c] == opponent:
        opponent_edges += 1
      if chess_board[board_size - 1, c] == player:
        player_edges += 1
      elif chess_board[board_size - 1, c] == opponent:
        opponent_edges += 1
    # Count pieces on left and right edges (excluding corners to avoid double counting)
    for r in range(1, board_size - 1):
      if chess_board[r, 0] == player:
        player_edges += 1
      elif chess_board[r, 0] == opponent:
        opponent_edges += 1
      if chess_board[r, board_size - 1] == player:
        player_edges += 1
      elif chess_board[r, board_size - 1] == opponent:
        opponent_edges += 1
    edge_advantage = player_edges - opponent_edges
    
    # Evaluate center control (center positions can influence more squares)
    center_weight = 0
    center_start = board_size // 3
    center_end = board_size - center_start
    for r in range(center_start, center_end):
      for c in range(center_start, center_end):
        if chess_board[r, c] == player:
          center_weight += 1
        elif chess_board[r, c] == opponent:
          center_weight -= 1
    
    # Combine all factors with weights
    # Material (piece count) is most important
    score = piece_difference * 100
    # Mobility is important - more moves means more options
    score += mobility_difference * 10
    # Corner control is valuable
    score += corner_advantage * 25
    # Edge control is moderately valuable
    score += edge_advantage * 5
    # Center control helps in early/mid game
    score += center_weight * 8
    
    return score
