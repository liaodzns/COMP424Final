# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

""""
MCTS node class models each node in the MCTS.
MCTS searches the state space of the Ataxx game modelled as a Markov Decision Process
(S, s_0 in S, A_s, P_a(s, s')), where 
- S is the state space, 
- s_0 is the initial state
- A_s is the set of actions which we can perform at state s in S
- P_a(s, s') is the probability transition function from state s to s' upon applying a to s. 
  Since Ataxx is deterministic, we transition a(s) = s' with probability 1.

Inputs:
    int array state: is the board state
    int player: is the current player, 1: Blue, 2: Brown
    MCTSNode parent: is None for the root node and the respective parent p for all other nodes s
    MCTSNode parent_action: is None for the root and the action a in A_p taken by the parent p of s for all other nodes s

Other fields:
    MCTS array children: stores the children of this MCTS Node
    int number_of_visits: is the total number of times this node has been visited
    int wins: is a mapping of the number of wins (incremented by one each win) achieved by the node over all trials
    MoveCoordinates array untried_actions: is an array of the legal actions that we have not yet applied to this node    

References:  
- https://arxiv.org/pdf/2103.04931
- https://ai-boson.github.io/mcts/    
"""
class MCTSNode():
    def __init__(self, state, player, parent=None, parent_action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.wins = 0
        self.untried_actions = get_valid_moves(self.state, self.player)

    # Expands the selected non-terminal node over all applicable actions and appends the children 
    def expand(self):
      action = self.untried_actions.pop()
      next_state = execute_move(deepcopy(self.state), action, self.player)
      next_player = 3 - self.player # Switch between player 1 and player 2
      child = MCTSNode(next_state, player=next_player, parent=self, parent_action=action)
      self.children.append(child)
      return child
    
    # Returns true or false if a given node is terminal or not
    def is_terminal_node(self):
      return check_endgame(self.state)[0]
    
    # Simulates the entire game from the current state following the rollout policy until a terminal node is reached.
    # Then returns the outcome utility of the simulation.
    def rollout(self):
       curr_rollout_state = deepcopy(self.state)
       curr_player = self.player
       is_endgame, p0_score, p1_score = check_endgame(curr_rollout_state)
       while not is_endgame:
        legal_actions = get_valid_moves(curr_rollout_state, curr_player)
        action = self.default_policy(legal_actions)
        curr_rollout_state = execute_move(curr_rollout_state, action, curr_player)
        curr_player = 3 - curr_player  # Switch player
        is_endgame, p0_score, p1_score = check_endgame(curr_rollout_state)

       if self.player == 1:
         return int(p0_score > p1_score)
       else:
         return int(p1_score > p0_score)

    # Backpropagates the results of simulation to each node
    def backpropagate(self, utility):
      self.number_of_visits += 1
      self.wins += utility
      if self.parent:
        self.parent.backpropagate(utility) # recursive call

    def q(self):
      return self.wins

    def n(self):
      return self.number_of_visits
    
    # Tree policy uses UCT to select best child to expand
    def best_child(self, C=np.sqrt(2)):
      Qs = [(child.q() / child.n()) + C * np.sqrt(np.log(self.n() / child.n())) for child in self.children]
      return self.children[np.argmax(Qs)]
      
    # Default policy: randomly selects an action to take
    def default_policy(self, legal_actions):
      return legal_actions[np.random.randint(len(legal_actions))]

    def is_fully_expanded(self):
      return len(self.untried_actions) == 0

    # Expands the current node and the selects the best child to rollout
    def tree_policy(self):
      curr_node = self
      while not curr_node.is_terminal_node():
        if not curr_node.is_fully_expanded():
          return curr_node.expand()
        else:
          curr_node = curr_node.best_child()
      return curr_node
    
    def get_parent_action(self):
      return self.parent_action

    # Finds the best move in n simulations via the four step MCTS process: Select -> Expand -> Simulate -> Backpropagate
    # Simulates until just before the time limit (.05 seconds before time limit of 2 by default)
    def best_action(self, time_limit=1.95):
      start_time = time.time()
      while True:
        if time.time() - start_time > time_limit:
          break
        v = self.tree_policy()
        wins = v.rollout()
        v.backpropagate(wins)
      return self.best_child().get_parent_action()


@register_agent("mcts_agent")
class MCTSAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(MCTSAgent, self).__init__()
    self.name = "MCTSAgent"

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

    root = MCTSNode(state=chess_board, player=player)
    best_move = root.best_action()

    if best_move is None:
      return random_move(chess_board, player)
    return best_move