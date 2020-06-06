import numpy as np

# returns a new random graph with n nodes numbered 0 to n−1 such that every 
# different pair of nodes is connected with probability p. Assume n>1, and 0≤p≤1.
# ie: {0: [2], 1: [2, 4], 2: [0, 1, 4], 3: [], 4: [1, 2]}
def rand_graph(p, n):
  
    friends = list(range(n))
    graph = {k: [] for k in friends}
    for i, friend in enumerate(friends,0):
        size = len(friends)-i-1
        probability_list = [p]*size

        # get relationship with every person that has not been explored. 1=friend 0= notafriend
        results = np.random.binomial(1, probability_list) 

        for j, result in enumerate(results,i+1):
            if result == 1:
                list_f = graph[friend] # get list of current friends
                list_f.append(j) # add to list
                graph[friend] = list_f # update friend list 
                
                list_f_f = graph[j] # get list of friend's friends
                list_f_f.append(friend) # add to list
                graph[j] = list_f_f # update friend's friend list
    return graph
                
# rand_graph(0.1,5)

