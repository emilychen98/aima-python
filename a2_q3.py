from a2_q1 import *
from a2_q2 import *
from csp import *
import time

NUM_ASSIGNED = 0
NUM_UNASSIGNED = 0

# number of teams, running time (seconds), # of assigned CSP variables # of unassigned CSP variables, ____________
solution_results = [] 

# ______________________________________________________________________________

def run_q3():
    graphs = [rand_graph(0.1, 31), rand_graph(0.2, 31), rand_graph(0.3, 31),
              rand_graph(0.4, 31), rand_graph(0.5, 31), rand_graph(0.6, 31)]

    groupNames = list(range(31))

    for graph in graphs:
        start_time = time.time()

        csp = MapColoringCSP(groupNames, graph)
        result = backtracking_search(csp)
        teams = set(result.values())
        numTeams = len(teams) # number of teams that the people are divided into

        elapsed_time = time.time() - start_time # running time of the solver

        print("Number of teams that the people are divided into: %d"%(numTeams))
        print("Running time of the solver: %f"%(elapsed_time))
        print("Number of times CSP variables were assigned: %d"%(NUM_ASSIGNED))
        print("Number of times CSP variables were unassigned: %d"%(NUM_UNASSIGNED))
        print("???\n")

        # Variable to hold solution results to be processed in a .csv file
        solution_results.append([numTeams, elapsed_time, NUM_ASSIGNED, NUM_UNASSIGNED])

# ______________________________________________________________________________

def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                global NUM_ASSIGNED
                NUM_ASSIGNED +=1
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        global NUM_UNASSIGNED
        NUM_UNASSIGNED += 1
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


# Test Question 3
# ______________________________________________________________________________
for i in range(7): 
    run_q3() 

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
