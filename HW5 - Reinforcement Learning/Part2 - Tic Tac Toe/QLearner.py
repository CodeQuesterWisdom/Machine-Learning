from Board import Board
import numpy as np
np.random.seed(10)
class QLearner:
    """  Your task is to implement `move()` and `learn()` methods, and
         determine the number of games `GAME_NUM` needed to train the qlearner
         You can add whatever helper methods within this class.
    """
    # ======================================================================
    # ** set the number of games you want to train for your qlearner here **
    # ======================================================================
    GAME_NUM = 75000


    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        self.alpha = 0.1
        self.gamma = 0.9
        self.states = []  # For storing each state in a game
        self.Q1= {}  # Q-learning for player1
        self.Q2= {}  # Q-learning for player1
        self.Q= {}

    def move(self, board):
        """ given the board, make the 'best' move
            currently, qlearner behaves just like a random player
            see `play()` method in TicTacToe.py
        Parameters: board
        """
        if board.game_over():
            return

        # find all legal moves
        candidates = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board.is_valid_move(i, j):
                    candidates.append(tuple([i, j]))


        # Load corresponding Q dictionary based on player1 or player2
        me = self.side
        if me==1: self.Q = self.Q1
        elif me==2:  self.Q = self.Q2

        # Encoding current board state
        curr_state = board.encode_state()

        # Greedy Approach to pick a move from legal moves

        # If the Q-learner doesn't have the current state recorded
        # Pick some random move from valid moves
        if curr_state not in self.Q:
            indice = np.random.randint(len(candidates))
            move = candidates[indice]

        # Else Greedily choose the move which has highest Q value
        else:
            values= []

            # Obtain Q values for all valid moves
            for candidate in candidates:
                if candidate in self.Q[curr_state]:
                    values.append(self.Q[curr_state][candidate])
                else:
                    values.append(0)

            values = np.array(values)

            # Find location of maximum value
            indice = np.where(values == np.max(values))[0]

            # If more than one maximum actions are present, choose one randomly from them
            if len(indice) > 1:
                index_selected = np.random.choice(indice, 1)[0]

            # If only one max action is present, choose that action
            else:
                index_selected = indice[0]
            move = candidates[index_selected]

        # Record the current state and action choosen
        self.states.append([curr_state,move])

        return board.move(move[0], move[1], self.side)


    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py
        Parameters: board
        """

        me = self.side  # To know the player

        # Load corresponding Q dict based on player
        if me==1: self.Q = self. Q1
        elif me==2:  self.Q = self.Q2

        # Game result
        res = board.game_result

        # Assign Rewards
        if res == 0: reward = 0
        elif me==1 and res == 1: reward = 20
        elif me==2 and res == 2: reward = 20
        elif me==1 and res == 2: reward = -20
        elif me==2 and res == 1: reward = -20

        count= 1

        # Q-Iteration - to update Q values
        for st,action in reversed(self.states):
            if st not in self.Q:
                self.Q[st] = {}
                self.Q[st][action] = 0

            # Terminal case
            if count==1:
                # Utility value of terminal state is it's reward
                self.Q[st][action] = reward
                # Storing current action which will be next action for previous state
                next_action = self.Q[st][action]

            # Non-Terminal case
            else:
                if action not in self.Q[st]: self.Q[st][action] = 0
                # Reward is zero for all other states
                self.Q[st][action] = (1 - self.alpha)* self.Q[st][action] + (self.alpha * self.gamma * next_action)
                # Storing current action which will be next action for previous state
                next_action = self.Q[st][action]

            count+=1

        # Emptying state after the game is done
        self.states = []

        # Updating corresponding Q dict
        if me==1: self.Q1 = self.Q
        elif me==2:  self.Q2 = self.Q


    # do not change this function
    def set_side(self, side):
        self.side = side
