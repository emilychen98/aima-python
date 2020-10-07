
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
