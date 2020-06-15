from a2_q1 import *
from a2_q2 import *
from csp import *
import time

TEST_COUNTER = 1

# number of teams, running time (seconds), # of assigned CSP variables # of unassigned CSP variables, ____________
solution_results = [] 

TOTAL_ASSIGNS = 0
TOTAL_UNASSIGNS = 0

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
        self.prune_count = 0 # of times prune function is called

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
        self.prune_count += 1
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

def run_q3():
    graphs = [rand_graph(0.1, 31), rand_graph(0.2, 31), rand_graph(0.3, 31),
              rand_graph(0.4, 31), rand_graph(0.5, 31), rand_graph(0.6, 31)]

    global TEST_COUNTER
    print("RUNNING TEST RUN #%d"%(TEST_COUNTER))
    print("====================================================================================\n\n")

    for graph in graphs:
        global TOTAL_ASSIGNS
        global TOTAL_UNASSIGNS
        TOTAL_ASSIGNS = 0
        TOTAL_UNASSIGNS = 0
        result = not None
        start_time = time.time()
        for i in range(31, 0, -1): #countdown from 31 to 1     
            groupNames = list(range(i))
            csp = MapColoringCSP(groupNames, graph)
            solvable = AC3(csp)

            if not solvable:
                continue
            result = backtracking_search(csp, select_unassigned_variable=mrv, inference=forward_checking)
            TOTAL_ASSIGNS+=csp.nassigns
            TOTAL_UNASSIGNS+=csp.unassigns
            if result == None:
                # we have found the largest # of teams that is not satisfiable. Increment counter to get 
                # largest satisfiable team for the problem.
                groupNames = list(range(i+1))
                csp = MapColoringCSP(groupNames, graph)
                AC3(csp)
                result = backtracking_search(csp, select_unassigned_variable=mrv, inference=forward_checking)
                TOTAL_ASSIGNS+=csp.nassigns
                TOTAL_UNASSIGNS+=csp.unassigns
                break
       
        elapsed_time = time.time() - start_time # running time of the solver

        teams = set(result.values())
        numTeams = len(teams) # number of teams that the people are divided into
        check_teams_result=check_teams(graph, result)

        print("Number of teams that the people are divided into: %d"%(numTeams))
        print("Running time of the solver: %f"%(elapsed_time))
        print("Number of times CSP variables were assigned for %d teams: %d"%(numTeams, csp.nassigns))
        print("Number of times CSP variables were unassigned for %d teams: %d"%(numTeams, csp.unassigns))
        print("Total Number of times CSP variables were assigned during the duration of the solver: %d"%(TOTAL_ASSIGNS))
        print("Total Number of times CSP variables were unassigned during the duration of the solver: %d"%(TOTAL_UNASSIGNS))
        print("Number of times the prune() function is called: %d"%(csp.prune_count))
        print("Check generated teams results: %s\n"%(check_teams_result))

        # Variable to hold solution results to be processed in a .csv file
        solution_results.append([str(graph), numTeams, elapsed_time, csp.nassigns, csp.unassigns, TOTAL_ASSIGNS, TOTAL_UNASSIGNS, csp.prune_count])

    TEST_COUNTER+=1

# Test Question 3
# ______________________________________________________________________________
for i in range(5): 
    run_q3() 

# Write data to csv file for processing
# _____________________________________________________________________
f2 = open('q3.csv', 'w')
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
