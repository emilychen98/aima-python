#min_conflicts function in CSP
from a2_q1 import *
from a2_q2 import *
from csp import *
import time

TEST_COUNTER = 1

# number of teams, running time (seconds), # of assigned CSP variables # of unassigned CSP variables, ____________
solution_results = [] 

# total number of conflicts in min_conflicts 
NUM_CONFLICTS = 0

# ______________________________________________________________________________
class CSP(search.Problem):
    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0
        self.unassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]
            self.unassigns += 1

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]

def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors, different_values_constraint)


# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=2000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    global NUM_CONFLICTS
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        NUM_CONFLICTS +=1
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))
# ______________________________________________________________________________

def run_q4():
    graphs = [rand_graph(0.1, 105), rand_graph(0.2, 105), rand_graph(0.3, 105),
              rand_graph(0.4, 105), rand_graph(0.5, 105), rand_graph(0.6, 105)]

    global TEST_COUNTER
    print("RUNNING TEST RUN #%d"%(TEST_COUNTER))
    print("====================================================================================\n\n")

    for graph in graphs:
        result = not None
        start_time = time.time()
        global NUM_CONFLICTS
        NUM_CONFLICTS = 0
        for i in range(1, 106, 1): #count up from 1 to 105
            groupNames = list(range(i))
            csp = MapColoringCSP(groupNames, graph)
            AC3(csp)
            result = min_conflicts(csp)
            if result != None:
                break
        elapsed_time = time.time() - start_time

        teams = set(result.values())
        numTeams = len(teams) # number of teams that the people are divided into
        check_teams_result=check_teams(graph, result)

        print("Number of teams that the people are divided into: %d"%(numTeams))
        print("Running time of the solver: %f"%(elapsed_time))
        print("Number of times CSP variables were assigned: %d"%(csp.nassigns))
        print("Number of times CSP variables were unassigned: %d"%(csp.unassigns))
        print("Number of attempts to resolve conflicts: %d"%(NUM_CONFLICTS))
        print("Check generated teams results: %s\n"%(check_teams_result))

        # Variable to hold solution results to be processed in a .csv file
        solution_results.append([str(graph), numTeams, elapsed_time, csp.nassigns, csp.unassigns, NUM_CONFLICTS])

    TEST_COUNTER+=1

# Test Question 4
# _____________________________________________________________________
for i in range(5): 
    run_q4() 

# Write data to csv file for processing
# _____________________________________________________________________
f2 = open('q4.csv', 'w')
for r in solution_results:
    for c in r:
        if type(c) == str:
            f2.write('%s;' %c)
        elif type(c) == int:
            f2.write('%d;' %c)
        else:
            f2.write('%f;' %c)
    f2.write('\n')
f2.close()
