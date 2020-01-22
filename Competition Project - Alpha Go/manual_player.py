import sys

class ManualPlayer():
    def __init__(self):
        self.type = 'manual'
        self.input_hint = True

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''    
        sign = 'X' if piece_type == 1 else 'O'
        print(('Turn for {}. Please make a move.').format(sign))

        # Pass input
        while 1:
            if self.input_hint: 
                print('Input format: row, column. E.g. 2,3')
                self.input_hint = False

            user_input = input('Input:')
            if user_input.lower() == 'exit':
                sys.exit()
            try:
                input_coordinates = user_input.strip().split(',')
                i, j = int(input_coordinates[0]), int(input_coordinates[1])
                return i, j
            except:
                print('Invalid input. Input format: row, column. E.g. 2,3\n')
                self.input_hint = True
                continue