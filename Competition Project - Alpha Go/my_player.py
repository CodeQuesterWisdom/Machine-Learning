import sys
import json
import ast
import numpy as np
#np.random.seed(10)

class MyPlayer():
    def __init__(self):
        self.type = 'my'
        self.alpha = 0.2
        self.gamma = 0.9
        self.states = []  # For storing each state in a game
        self.Q1= {}  # Q-learning for player1
        self.Q2= {}  # Q-learning for player1
        with open("dict1.txt","r") as f:
            json_string = f.read()
            if json_string:
                self.Q1 = ast.literal_eval(json_string)

        with open("dict2.txt","r") as f:
            json_string = f.read()
            if json_string:
                self.Q2 = ast.literal_eval(json_string)

        self.Q= {}
        self.eps= 0

    def encode_state(self,board,BOARD_SIZE):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(board[i][j]) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''

        # find all legal moves

        candidates = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    candidates.append((i,j))

        if np.random.uniform(0,1)< self.eps:
            indice = np.random.randint(len(candidates))
            move = candidates[indice]
            curr_state = self.encode_state(go.board,go.size)
        else:
        # Load corresponding Q dictionary based on player1 or player2
            me = piece_type
            if me==1: self.Q = self.Q1
            elif me==2:  self.Q = self.Q2

            # Encoding current board state
            curr_state = self.encode_state(go.board,go.size)

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

        return (move[0], move[1])



    def learn(self,result,piece_type,my_go,time):
        me = piece_type  # To know the player

        # Load corresponding Q dict based on player
        if me==1: self.Q = self. Q1
        elif me==2:  self.Q = self.Q2

        # Game result
        res = result

        # Assign Rewards
        if res == 0: reward = 0
        elif me==1 and res == 1: reward = 50
        elif me==2 and res == 2: reward = 50
        elif me==1 and res == 2: reward = -50
        elif me==2 and res == 1: reward = -50

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
                self.Q[st][action] = ((1 - self.alpha)* self.Q[st][action]) + (self.alpha * (next_action + (self.gamma * self.Q[st][action])))
                # Storing current action which will be next action for previous state
                next_action = self.Q[st][action]

            count+=1

        # Emptying state after the game is done
        self.states = []

        # Updating corresponding Q dict
        if me==1: self.Q1 = self.Q
        elif me==2:  self.Q2 = self.Q


        if time%100 == 0:

            print("loop",time)
            print("dict1",len(self.Q1))
            print("dict2",len(self.Q2))
            print("-------------------")

            f = open("dict1.txt","w")
            f.write(str(self.Q1))
            f.close()

            f = open("dict2.txt","w")
            f.write(str(self.Q2))
            f.close()

            with open("dict1.txt","r") as f:
                json_string = f.read()
                if json_string:
                    self.Q1 = ast.literal_eval(json_string)

            with open("dict2.txt","r") as f:
                json_string = f.read()
                if json_string:
                    self.Q2 = ast.literal_eval(json_string)
