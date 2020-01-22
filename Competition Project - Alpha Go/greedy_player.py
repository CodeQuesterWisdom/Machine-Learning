import random
import sys

class GreedyPlayer():
    def __init__(self):
        self.type = 'greedy'

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''    
        largest_died_chess_cnt = 0
        greedy_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    test_go = go.copy_board()
                    test_go.place_chess(i, j, piece_type)
                    died_chess_cnt = len(test_go.find_died_pieces(3 - piece_type))
                    if died_chess_cnt == largest_died_chess_cnt:
                        greedy_placements.append((i,j))
                    elif died_chess_cnt > largest_died_chess_cnt:
                        largest_died_chess_cnt = died_chess_cnt
                        greedy_placements = [(i,j)]
        return random.choice(greedy_placements)