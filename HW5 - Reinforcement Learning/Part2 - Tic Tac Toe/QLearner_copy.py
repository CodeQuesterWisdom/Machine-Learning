from Board import Board
import numpy as np
import copy

class QLearner:
    """  Your task is to implement `move()` and `learn()` methods, and
         determine the number of games `GAME_NUM` needed to train the qlearner
         You can add whatever helper methods within this class.
    """
    # ======================================================================
    # ** set the number of games you want to train for your qlearner here **
    # ======================================================================
    GAME_NUM = 50000


    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        self.alpha = 0.1
        self.gamma = 0.9
        self.exp_rate = 0.1
        #self.actions = []
        self.states = []
        # for i in range(3):
        #     for j in range(3):
        #         self.actions.append((i,j))
        self.Q_1= {}
        self.Q_2= {}
        self.Q= {}
        # for action in self.actions:
        #     self.Q[action] = 0


    # def getHash(self,board):
    #     self.boardHash = str(board.reshape(3 * 3))
    #     return self.boardHash

    def move(self, board):
        """ given the board, make the 'best' move
            currently, qlearner behaves just like a random player
            see `play()` method in TicTacToe.py
        Parameters: board
        """
        if board.game_over():
            return

        # =========================================================
        # ** Replace Your code here  **

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

        # if np.random.uniform(0, 1) <= self.exp_rate:
        if np.random.random()<= self.exp_rate:
            indice = np.random.randint(len(candidates))
            move = candidates[indice]
        else:
            # Greedy Approach
            values = []
            for p in candidates:
                next_board = copy.deepcopy(board)
                next_board.state[p[0]][p[1]] = self.side
                next_boardHash = next_board.encode_state()
                if next_boardHash in self.Q:
                    values.append(self.Q[next_boardHash])
                else:
                    values.append(-float("inf"))
            values = np.array(values)
            indice = np.where(values == np.max(values))[0]
            if len(indice) > 1:
                index_selected = np.random.choice(indice, 1)[0]
            else:
                index_selected = indice[0]
            move = candidates[index_selected]

        next_board = copy.deepcopy(board)
        next_board.state[move[0]][move[1]] = self.side
        next_boardHash = next_board.encode_state()
        self.states.append(next_boardHash)

        # randomly select one and apply it
        # idx = np.random.randint(len(candidates))
        # move = candidates[idx]
        # =========================================================

        return board.move(move[0], move[1], self.side)


    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py
        Parameters: board
        """

        me = self.side
        res = board.game_result

        if res == 0: reward = 0
        elif me==1 and res == 1: reward = 1
        elif me==2 and res == 2: reward = 1
        elif me==1 and res == 2: reward = -1
        elif me==2 and res == 1: reward = -1

        for st in reversed(self.states):
            if st not in self.Q:
                self.Q[st] = 0
            #self.Q[st] = (1 - self.alpha)* self.Q[st] + self.alpha * (reward + self.d*max(Q_options))
            self.Q[st] = (1 - self.alpha)* self.Q[st] + self.alpha * (reward  * self.gamma)
            reward = self.Q[st]

        self.states = []

    # do not change this function
    def set_side(self, side):
        self.side = side
