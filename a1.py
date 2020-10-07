# a1.py

from search import *
import time
import random
import numpy as np
import csv

puzzle_results = []
single_result = []

"""returns a new instance of an EightPuzzle problem with a random initial state that is solvable. 
   Note that EightPuzzle has a method called check_solvability that you should use to help ensure 
   your initial state is solvable."""
def make_rand_8puzzle():
    puzzle = tuple(np.random.permutation(9))
    eightPuzzle_Instance = EightPuzzle(puzzle)
    solvable = eightPuzzle_Instance.check_solvability(eightPuzzle_Instance.initial)
    while not solvable:
        puzzle = tuple()
        puzzle = tuple(np.random.permutation(9))
        eightPuzzle_Instance = EightPuzzle(puzzle)
        solvable = eightPuzzle_Instance.check_solvability(eightPuzzle_Instance.initial)
       
    return EightPuzzle(puzzle)

"""returns a new instance of an DuckPuzzle problem with a random initial state that is solvable. """
def make_rand_duck_puzzle():
    state = (1,2,3,4,5,6,7,8,0) # start from solvable state
    duckPuzzle_Instance = DuckPuzzle(state)
    new_state = list(state)
    
    for i in range(1000):
        possibleMoves = duckPuzzle_Instance.actions(state)
        blank = duckPuzzle_Instance.find_blank_square(state)
       
        delta = {}
        state1 = [2, 3]
        state2 = [0, 1]
        if blank in state1:
            delta = {'UP': -2, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1} 

        elif blank in state2:
            delta = {'UP': -3, 'DOWN': 2, 'LEFT': -1, 'RIGHT': 1} 
        else:
            delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}

        neighbor = blank + delta[random.choice(possibleMoves)]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]
        duckPuzzle_Instance = DuckPuzzle(new_state)
        state = new_state

    new_state = tuple(new_state)
    return DuckPuzzle(new_state)

"""takes an 8-puzzle state (i.e. a tuple that is a permutation of (0, 1, 2, …, 8)) as input and prints 
   a neat and readable representation of it. 0 is the blank, and should be printed as a * character."""
def display(state):
    display_string = ""
    for i, value in enumerate(state, 1):
        if value == 0:
            value = '*'
        if i % 3 == 0:
            display_string += str(value)
            display_string += "\n"
        else:
            display_string += str(value) + " "
    
    print(display_string + '\n')

"""takes an 8-puzzle state (i.e. a tuple that is a permutation of (0, 1, 2, …, 8)) as input and prints 
   a neat and readable representation of it in a Duck Display. 0 is the blank, and should be printed as a * character."""
def displayDuck(state):
    display_string = ""
    for i, value in enumerate(state, 1):
        if value == 0:
            value = '*'
        if i == 3:
            display_string += "\n" + str(value) + " "
        elif i == 7:
            display_string += "\n" + "  " + str(value) + " "
        else:
            display_string += str(value) + " "
    print(display_string + '\n')

# ______________________________________________________________________________
# Informed (Heuristic) Search

# h=None (misplaced tile heuristic)
# h=h_manhattan (manhattan distance heuristic)
# h=h_max_manhattan_or_misplaced (max of miplaced tile heuristic and manhattan distance heuristic)
def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    start_time = time.time()
    h = memoize(h or problem.h, 'h')
    result = best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)
    elapsed_time = time.time() - start_time

    actions = result[0].solution() # result[0] = node
    length = len(actions)
    nodesRemoved = result[1] # result[1] = counter of # of frontiers popped

    print("Total Running Time in Seconds: %f"%(elapsed_time))
    print("Length of Solution (Number of Tiles Moved): %d"%(length))
    print("Total Number of Nodes Removed From Frontier: %d\n"%(nodesRemoved))

    # The code below was used to store data so that it will be ready to print
    # to a csv file for data processing. See the commented out code at the end
    # of this file for more info
    single_result.append(elapsed_time)
    single_result.append(length)
    single_result.append(nodesRemoved)

    # All 3 heuristics have been run on a single state. Add results to row
    if len(single_result) == 10:
        puzzle_results.append(single_result)


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node, len(explored)+1
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def h_misplaced(node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        initial_state = node.state
        goal_state = (1,2,3,4,5,6,7,8,0)

        count = 0;
        for i in range(0,9):
            if initial_state[i] == 0:
                continue;
            if initial_state[i] != goal_state[i]:
                count = count +1
        return count

def h_manhattan(node):
    initial_state = node.state
    goal_state = (1,2,3,4,5,6,7,8,0)

    initial_state_array = [ [initial_state[0], initial_state[1], initial_state[2]], [initial_state[3], initial_state[4], initial_state[5]], [initial_state[6], initial_state[7], initial_state[8]] ]
    goal_state_array = [ [goal_state[0], goal_state[1], goal_state[2]], [goal_state[3], goal_state[4], goal_state[5]], [goal_state[6], goal_state[7], goal_state[8]] ]

    initial_state_array = np.array(initial_state_array)
    goal_state_array = np.array(goal_state_array)
   
    manhattan_Distance = 0
    for i in range(1,9):
        i_coord = np.argwhere(initial_state_array == i)
        i_x = i_coord[0][0] # row index
        i_y= i_coord[0][1] # col index

        y_coord = np.argwhere(goal_state_array == i)
        g_x = y_coord[0][0] # row index
        g_y= y_coord[0][1] # col index

        manhattan_Distance = manhattan_Distance + abs(i_x-g_x)+abs(i_y-g_y)

    return manhattan_Distance

def h_duck_manhattan(node):
    initial_state = node.state
    goal_state = (1,2,3,4,5,6,7,8,0)

    initial_state_dict = {
        initial_state[0]:[0,0],
        initial_state[1]:[0,1],
        initial_state[2]:[1,0],
        initial_state[3]:[1,1],
        initial_state[4]:[1,2],
        initial_state[5]:[1,3],
        initial_state[6]:[2,1],
        initial_state[7]:[2,2],
        initial_state[8]:[2,3]
    }

    goal_state_dict = {
        goal_state[0]:[0,0],
        goal_state[1]:[0,1],
        goal_state[2]:[1,0],
        goal_state[3]:[1,1],
        goal_state[4]:[1,2],
        goal_state[5]:[1,3],
        goal_state[6]:[2,1],
        goal_state[7]:[2,2],
        goal_state[8]:[2,3]
    }

    manhattan_Distance = 0
    for i in range(1,9):
        initial = initial_state_dict[i]
        goal = goal_state_dict[i]

        initial_x = initial[0] # row index
        initial_y= initial[1] # col index

        goal_x = goal[0] # row index
        goal_y= goal[1] # col index

        manhattan_Distance = manhattan_Distance + abs(initial_x-goal_x)+abs(initial_y-goal_y)

    return manhattan_Distance

def h_max_manhattan_or_misplaced(node):
    num_misplaced = h_misplaced(node)
    manhattan_distance = h_manhattan(node)
    if manhattan_distance > num_misplaced:
        return manhattan_distance
    else:
        return num_misplaced

def h_max_duck_manhattan_or_misplaced(node):
    num_misplaced = h_misplaced(node)
    manhattan_distance = h_duck_manhattan(node)
    if manhattan_distance > num_misplaced:
        return manhattan_distance
    else:
        return num_misplaced

# ______________________________________________________________________________
# A* heuristics 
class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        initial_state = node.state
        goal_state = (1,2,3,4,5,6,7,8,0)

        count = 0;
        for i in range(0,9):
            if initial_state[i] == 0:
                continue;
            if initial_state[i] != goal_state[i]:
                count = count +1
        return count


class DuckPuzzle(Problem):
    
    """    +--+--+
           |  |  |
           +--+--+--+--+
           |  |  |  |  |
           +--+--+--+--+
              |  |  |  | 
              +--+--+--+    """
   
    """ The problem of sliding tiles numbered from 1 to 8 on a board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        removeLeft = [0, 2, 6]
        removeUp = [0, 1, 4, 5]
        removeRight = [1, 5, 8]
        removeDown = [2, 6, 7, 8]

        if index_blank_square in removeLeft:
            possible_actions.remove('LEFT')
        if index_blank_square in removeUp:
            possible_actions.remove('UP')
        if index_blank_square in removeRight:
            possible_actions.remove('RIGHT')
        if index_blank_square in removeDown:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {}
        state1 = [2, 3]
        state2 = [0, 1]
  
        if blank in state1:
            delta = {'UP': -2, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1} 

        elif blank in state2:
            delta = {'DOWN': 2, 'LEFT': -1, 'RIGHT': 1} 
        else:
            delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}

        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        initial_state = node.state
        goal_state = (1,2,3,4,5,6,7,8,0)

        count = 0;
        for i in range(0,9):
            if initial_state[i] == 0:
                continue;
            if initial_state[i] != goal_state[i]:
                count = count +1
        return count

# ______________________________________________________________________________

# Test Program #1
#""" Create 10 (more would be better!) random 8-puzzle instances (using your code from above), and solve each of 
#them using the algorithms. Each algorithm is run on the exact same set of problems to make the comparison fair."""

print("====================================================================================\n")
print("-----------------------------RUNNING EIGHT PUZZLE TESTS -----------------------------\n")
print("====================================================================================\n\n\n")

state_List = []
for i in range (10):
    state_List.append(make_rand_8puzzle())

# Test Misplaced Heuristic
#print("Testing Misplaced Heuristic")
#print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    display(state.initial)
    astar_search(EightPuzzle(state.initial))

# Test Manhattan Distance Heuristic
print("\nTesting Manhattan Distance Heuristic")
print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    display(state.initial)
    astar_search(EightPuzzle(state.initial), h=h_manhattan)

# Test max of the misplaced tile heuristic and the Manhattan distance heuristic
print("\nTesting Max of Misplaced Tile and Manhattan Distance Heuristic")
print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    display(state.initial)
    astar_search(EightPuzzle(state.initial), h=h_max_manhattan_or_misplaced)


# Test Program #2
#""" Create 10 (more would be better!) random 8-puzzle instances (using your code from above), and solve each of 
#them using the algorithms. Each algorithm is run on the exact same set of problems to make the comparison fair."""

print("====================================================================================\n")
print("-----------------------------RUNNING DUCK PUZZLE TESTS -----------------------------\n")
print("====================================================================================\n\n\n")

state_List = []
for i in range (10):
    state_List.append(make_rand_duck_puzzle())

# Test Misplaced Heuristic
print("Testing Misplaced Heuristic")
print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    displayDuck(state.initial)
    astar_search(DuckPuzzle(state.initial))

# Test Manhattan Distance Heuristic
print("\nTesting Manhattan Distance Heuristic")
print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    displayDuck(state.initial)
    astar_search(DuckPuzzle(state.initial), h=h_duck_manhattan)

# Test max of the misplaced tile heuristic and the Manhattan distance heuristic
print("\nTesting Max of Misplaced Tile and Manhattan Distance Heuristic")
print("====================================================================================\n\n")

for i, state in enumerate(state_List):
    print("Test %d\n"%(i+1))
    displayDuck(state.initial)
    astar_search(DuckPuzzle(state.initial), h=h_max_duck_manhattan_or_misplaced)


# NOTE: The commented code below was used to write results to csv files for processing

#for i in range (10):
#    single_result = []
#    state = make_rand_8puzzle();
#    single_result.append(str(state.initial))
#    display(state.initial)

#    astar_search(EightPuzzle(state.initial))
#    astar_search(EightPuzzle(state.initial), h=h_manhattan)
#    astar_search(EightPuzzle(state.initial), h=h_max_manhattan_or_misplaced)

#f2 = open('results.csv', 'w')
#for r in puzzle_results:
#    for c in r:
#        if type(c) == str:
#            f2.write('%s;' %c)
#        elif type(c) == int:
#            f2.write('%d;' %c)
#        else:
#            f2.write('%f;' %c)
#    f2.write('\n')
#f2.close()

#for i in range (10):
#    single_result = []
#    state = make_rand_duck_puzzle();
#    single_result.append(str(state.initial))
#    displayDuck(state.initial)

#    astar_search(DuckPuzzle(state.initial))
#    astar_search(DuckPuzzle(state.initial), h=h_duck_manhattan)
#    astar_search(DuckPuzzle(state.initial), h=h_max_duck_manhattan_or_misplaced)

#f3 = open('results_duck.csv', 'w')
#for r in puzzle_results:
#    for c in r:
#        if type(c) == str:
#            f3.write('%s;' %c)
#        elif type(c) == int:
#            f3.write('%d;' %c)
#        else:
#            f3.write('%f;' %c)
#    f3.write('\n')
#f3.close()
