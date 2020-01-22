import sys
import collections


# Extracting data from Command Line Arguments
no_of_cli_arguments = len(sys.argv)
if no_of_cli_arguments!=3:
    print("Blackbox path not specified; Please check the CLI arguments")
    sys.exit(0)
args = sys.argv
input_file_path = args[1]
output_file = args[2]

# Extracting data from input file
f = open(input_file_path)
input_file = f.readlines()
input_file = [line.strip() for line in input_file]
grid_size=int(input_file[0])
no_of_obstacles= int(input_file[1])
obstacles = []  # type: List[Any]
destination = []
validStates = []

# Extracting obstacles from input_file
for i in range(2, 2 + no_of_obstacles):
    obstacle = input_file[i].split(",")
    x = int(obstacle[0])
    y = int(obstacle[1])
    obstacles.append((x,y))

# Extracting destination from input_file
curr_pos = 2 + no_of_obstacles
terminal = input_file[curr_pos].split(",")
x = int(terminal[0])
y = int(terminal[1])
destination.append((x,y))

# Defining stopping condition
gamma = 0.9
epsilon = 0.1
threshold = epsilon * (1 - gamma) / gamma

# Intializing board for maze
init_board={}
for i in range(grid_size):
    for j in range(grid_size):
        init_board[(j,i)] = 0

# It returns utility values from all four directions [N,S,E,W] of a given cell
def get_state_value(state,direction):
    global validStates
    if direction == "North":
        newState = (state[0], state[1]-1)
    elif direction == "South":
        newState = (state[0], state[1] + 1)
    elif direction == "East":
        newState = (state[0] + 1, state[1])
    elif direction == "West":
        newState = (state[0] - 1, state[1])

    if newState in validStates: return newState
    # If there is wall, return current cell value
    else: return state


# Core logic for updating policy values using value iteration
def calculate_utility(destination):
    global validStates

    # Get all possible positions of a maze
    validStates = init_board.keys()
    possibleDirections = ["North", "South", "East", "West"]

    grid_new = {}
    for state in validStates:
        grid_new[state] = 0

    k =1
    threshold_satisfied = False
    while True:
        threshold_satisfied = True

        # Storing updating grid in old grid
        grid_old = grid_new.copy()

        for state in validStates:

            # Do not update utility for destination
            if state not in destination:

                # Get utility values of all 4 direction for a given cell
                up = grid_old[get_state_value(state,"North")]
                down = grid_old[get_state_value(state, "South")]
                right = grid_old[get_state_value(state, "East")]
                left = grid_old[get_state_value(state, "West")]

                # probablities * utilities
                vup = ((0.7 * up) + (0.1 * right) + (0.1 * left) + (0.1 * down))
                vdown = ((0.7 * down) + (0.1 * right) + (0.1 * left) + (0.1 * up))
                vright = ((0.7 * right) + (0.1 * up) + (0.1 * left) + (0.1 * down))
                vleft = ((0.7 * left) + (0.1 * right) + (0.1 * up) + (0.1 * down))

                # Find maximum utility value among utility values of 4 directions
                max_next_utility = max(vup, vdown, vright, vleft)

                # Value iteration formula application:
                # current_state_utility = current_state_reward + gamma * max(prob * utility of next state)
                if state in obstacles:
                    utility = -101 + (gamma * max_next_utility)
                else:
                    utility = -1 + (gamma * max_next_utility)

                # Updating grid
                grid_new[state] = utility

        # Utility value for destination is its reward
        grid_new[destination] = 99

        # Check stopping condition for all cells in the grid
        for state in grid_new:
            if not abs(grid_new[state] - grid_old[state]) < threshold:
                 threshold_satisfied = False

        # If all cells in the maze satisfy the stopping condition, exit from method
        if threshold_satisfied: return grid_old
        k+=1


# Determine if there is wall/boundary
def wall(row,col):
    #[N,S,E,W]
    directions=["no","no","no","no"]
    if row== 0:
        directions[0]="yes"
    if row==grid_size-1:
        directions[1]="yes"
    if col==0:
        directions[3]="yes"
    if col==grid_size-1:
        directions[2]="yes"
    return directions

# For determing policy i.e. direction of a given cell based on utility values in all 4 directions
def find_direction(i,j,grid):
    #[N,S,E,W]
    determine_wall = wall(i,j)
    l = []

    if determine_wall[0]=="yes": l.append(grid[i][j])
    else: l.append(grid[i-1][j])

    if determine_wall[1]=="yes": l.append(grid[i][j])
    else: l.append(grid[i+1][j])

    if determine_wall[2]=="yes": l.append(grid[i][j])
    else: l.append(grid[i][j+1])

    if determine_wall[3]=="yes": l.append(grid[i][j])
    else: l.append(grid[i][j-1])

    high = max(l)
    return l.index(high)

# Do value iteration and update grid
grid  = calculate_utility(destination[0])
policy_values= [[grid[(i,j)] for i in range(grid_size)]for j in range(grid_size)]

# Get policy from updated grid obtained after value iteration
policy = [["" for i in range(grid_size)]for j in range(grid_size)]
arrows = ['^','v','>','<']
for i in range(grid_size):
    for j in range(grid_size):
        # If it is obstacle state, put x
        if (j,i) in obstacles:
            policy[i][j] = 'x'
        # If it is destination state, put .
        elif (j,i) in destination:
            policy[i][j] = '.'
        # Find direction of a given cell
        else:
            policy[i][j] = arrows[find_direction(i,j,policy_values)]


# Writing output to a file
with open(output_file, "w") as txt_file:
    l = len(policy)-1
    for index,line in enumerate(policy):
        if index!=l:
            txt_file.write("".join(line) + "\n")
        else:
            txt_file.write("".join(line))
