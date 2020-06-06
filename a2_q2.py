
# Returns True if the given CSP solution dictionary csp_sol satisfies all 
# the constraints in the friendship graph, and False otherwise.
def check_teams(graph, csp_sol):
    # iterate through graph
    for person in graph:
        friends = graph[person]
        for friend in friends:
            # person is on the same team as the friend
            if csp_sol[person] == csp_sol[friend]: 
                return False
    return True

#graph = {0: [1, 2], 1: [0], 2: [0], 3: []} 
#csp_sol = {0:0, 1:1, 2:1, 3:0} 
#check_teams(graph, csp_sol)

#   How CSP Class does it
# - Make instance of CSP and initialize to variables and domains
# - Feed the CSP solution to the CSP instance by calling one function
# - Call function conflicted_vars to check if teams are valid
