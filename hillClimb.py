import heapq
from networkSim import NetworkSim as ns

class HillClimb:
    #okay this one isn't really hill climb anymore, more like just pure bellman
    @staticmethod
    def hill_climb_with_bellman(graph, num=1, gamma=0.7, horizon=3, num_samples=5):
        seeded_set = set()
        (utility, action) = ns.state_value_function(graph, num, gamma, horizon, num_samples)

        #inefficiency carried over from previous hill climbing algo
        #print("Predicted utility gain:" + str(utility))

        for node_index in action:
            seeded_set.add(graph.nodes[node_index]['obj'])
        
        return seeded_set
        

    @staticmethod
    def hill_climb(graph, num=1):
        seeded_set = set()
        node_values = []

        #iterate thru graph to see nodes that can pick
        for node in graph:
            #arbitrarily chosen discount factor
            discount_factor = 0.5

            #replace this value with the value function
            value = 0

            #bootstrap solution to not double activate node under guaranteed activation
            if graph.nodes[node]['obj'].isActive():
                value -= 10

            if not graph.nodes[node]['obj'].canCascade():
                value += 1
            else:
                queue = [(node, 0)]
                visited_set = set()
                visited_set.add(node)
                while queue:
                    current_node, depth = queue.pop()                    
                    current_node_obj = graph.nodes[current_node]['obj']
                    #value of nodes decrease as u get further from the activated node
                    value += 1 * discount_factor ** depth

                    #extension -- different edge weights -- multiply discount factor by all edge weights on the path

                    #get the nodes this node can cascade to
                    if not current_node_obj.canCascade():
                        continue
                    for neighbor in graph.neighbors(current_node):
                        neighbor_obj = graph.nodes[neighbor]['obj']
                        if neighbor not in visited_set and neighbor_obj.canCascade():
                            visited_set.add(neighbor)
                            next_depth = depth + 1
                            queue.append((neighbor, next_depth))

            node_values.append((value, node))
        top_nodes = heapq.nlargest(num, node_values)
        selected_nodes = [node for value, node in top_nodes]

        for node in selected_nodes:
            seeded_set.add(graph.nodes[node]['obj'])
            #the actual activation of nodes has been moved to a separate function; this function now only returns nodes to activate

        #print(top_nodes)
        
        return seeded_set