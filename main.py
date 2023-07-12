import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# def intersection(lst1, lst2):
#     lst3 = [(x,y) for (x,y) in lst1 if (x,y) in lst2 or (y,x) in lst2]
#     return lst3


# def grouped(iterable, n):
#     "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
#     return zip(*[iter(iterable)]*n)


# def get_common_edges(cycles):
#     all_edges = [zip(nodes,(nodes[1:]+nodes[:1])) for nodes in cycles]
    
#     all_edges = list(map(lambda edge: list(edge),all_edges))
    
    
    
#     edges = []
    
#     for one_edges,two_edges in grouped(all_edges,2):
#         edges.append(intersection(one_edges,two_edges))
        
    
#     return edges

def add_known_voltages_to_nodes(G,gnd):
    
    n = len(G.nodes) - 1
    
    counter = 0
    
    
    node_indices = nx.get_node_attributes(G, "i")
    
    
    
    for i, node in enumerate(G.nodes):
        
        if i >= n:
            break
        
        edge_nodes = list(set(G.edges(node)))
        
        for edge_node in edge_nodes:
        
            edges = G.get_edge_data(edge_node[0],edge_node[1])
            
            
            
            for j,attr in edges.items():
                
                node1_idx = node_indices[edge_node[0]]
                
            
                if 'v' in attr and attr['end'] == gnd:
                
                    nx.set_node_attributes(G,{attr['start']:{'v':attr['v']}})
                    counter += 1
                    
    return counter
    
def merge_nodes(G, n1, n2,voltage):
    
    supernode_name = f"{n1}_{n2}"
    
    node_indices = nx.get_node_attributes(G, "i")
    
    node1_idx = node_indices[n1]
    
    node2_idx = node_indices[n2]
    
    G.add_node(supernode_name,supernode=True,v=voltage,node1_index=node1_idx,node2_index=node2_idx,node1=n1,node2=n2)
    node1_edges = G.edges(n1,data=True)
    
    node2_edges = G.edges(n2,data=True)
    
    key = 0
    
    for node1_edge in node1_edges:
        
        if node1_edge[1] == n2:
            continue
        
        G.add_edge(supernode_name,node1_edge[1])
        
        edge_attrs = node1_edge[2]
        
        edge_attrs['node'] = n1
        
        
        
        nx.set_edge_attributes(G, {(supernode_name,node1_edge[1],key):edge_attrs})
        
        key += 1
        
    G.remove_node(n1)
    
    key = 0
    
    for node2_edge in node2_edges:
        if node2_edge[1] == n1:
            continue
        G.add_edge(supernode_name,node2_edge[1])
        
        edge_attrs = node2_edge[2]
        
        edge_attrs['node'] = n2
        
        
        
        nx.set_edge_attributes(G, {(supernode_name,node2_edge[1],key):edge_attrs})
        
        key += 1
    
    
    G.remove_node(n2)
    
    # print(G.edges(supernode_name,data=True))
    
    
def adjust_for_supernodes(G,gnd):
    battery_edges = nx.get_edge_attributes(G,'v')
    
    for battery_edge,voltage in battery_edges.items():
        node1,node2,_ = battery_edge
        
        if node1 == gnd or node2 == gnd:
            continue
        
        merge_nodes(G,node1,node2,voltage)
        
        

    
        
        
        
        

def construct_nodal_equations(G):
    
    ground = 'v2'
    
    count_removed = add_known_voltages_to_nodes(G,ground)
    
    
    
    adjust_for_supernodes(G,ground)
    
    supernodes = nx.get_node_attributes(G, "supernode")
    
    supernodes_idxs1 = nx.get_node_attributes(G, "node1_index")
    
    supernodes_idxs2 = nx.get_node_attributes(G, "node2_index")
    
    supernode_nodes1 = nx.get_node_attributes(G, "node1")
    
    supernode_nodes2 = nx.get_node_attributes(G, "node2")
    
    
    
    n = len(G.nodes) - 1 - count_removed + len(supernodes)
    
    lhs = np.zeros((n,n))
    
    voltages = np.zeros(n)
    
    node_indices = nx.get_node_attributes(G, "i")
    
    node_voltages = nx.get_node_attributes(G, "v")
    
   
    supernode_loop_counter = -1
    
    
    
    for i, node in enumerate(G.nodes):
        
        if node == ground:
            continue
        
        
        
        if node in node_voltages and node_voltages[node] and not supernode:
            continue
        
        
        
        if i >= n:
            break
        
        edge_nodes = list(set(G.edges(node)))
        
        
        
        
        
        
        for edge_node in edge_nodes:
            
            
                 
                
        
            edges = G.get_edge_data(edge_node[0],edge_node[1])
            
            
            if edge_node[0] in supernodes or edge_node[1] in supernodes:
                
                supernode = edge_node[0] if edge_node[0] in supernodes else edge_node[1]
                
                
            
                
                supernode1_idx = supernodes_idxs1[supernode]
                supernode2_idx = supernodes_idxs2[supernode]
                lhs[supernode_loop_counter][supernode1_idx] += 1
                lhs[supernode_loop_counter][supernode2_idx] -= 1
                
                if edge_node[0] in supernodes:
                
                    voltages[supernode_loop_counter] = node_voltages[supernode] 
                
                supernode_node1 = supernode_nodes1[supernode]
                
                supernode_node2 = supernode_nodes2[supernode]
                
                supernode_loop_counter -= 1
                
                for j,attr in edges.items():
                    
                    node2_idx = node_indices[edge_node[1]] if supernode == edge_node[0] else node_indices[edge_node[0]]
                    
                    if edge_node[0] == supernode:
                    
                        if attr['node'] == supernode_node1:
                            lhs[i][supernode1_idx] += 1 / attr['r']
                        elif attr['node'] == supernode_node2:
                            lhs[i][supernode2_idx] += 1 / attr['r']
                            
                    else:
                        
                        supernode_to_node_edges = list(filter(lambda edge: edge[0] == supernode and edge[1] == edge_node[0],G.edges(supernode,data=True)))
                        
                        for supernode_to_node_edge in supernode_to_node_edges:
                            
                            
                            if supernode_to_node_edge[2]['node'] == supernode_node1:
                                lhs[i][supernode1_idx] += 1 / supernode_to_node_edge[2]['r']
                            elif supernode_to_node_edge[2]['node'] == supernode_node2:
                                lhs[i][supernode2_idx] += 1 / supernode_to_node_edge[2]['r']
                        
                        
                        
                
                # print(G.get_edge_data(edge_node[0],edge_node[1]))
                
                continue
            
            
            
            for j,attr in edges.items():
                
                node1_idx = node_indices[edge_node[0]]
                
                node2_idx = node_indices[edge_node[1]]
                
                node2_voltage = edge_node[1] in node_voltages
                
                
                
                if 'r' in attr:
                
                    lhs[i][node1_idx] += 1 / attr['r']
                    
                    if node2_voltage:
                        
                        voltages[i] += node_voltages[edge_node[1]] / attr['r']
                    if edge_node[1] != ground:
                     lhs[i][node2_idx] -= 1 / attr['r']
                    
               
                
                    
                    
    # print(np.linalg.solve(lhs,voltages))
    
    print(lhs,voltages)
                
        
    return lhs,voltages



G = nx.MultiGraph()


G.add_node('v1',i = 0)

G.add_node('v2', i = 3)

G.add_node('v3', i = 1)

G.add_node('v4', i = 2)


G.add_edge('v1', 'v2', r=5)



G.add_edge('v1', 'v2', r=10)

G.add_edge('v3', 'v4', v=40, start='v3',end = 'v4')

G.add_edge('v3','v1',r=20)

G.add_edge('v4', 'v2', r=15)



# T = nx.minimum_spanning_tree(G)

# print(G.get_edge_data('v1','v2'))

construct_nodal_equations(G)

node_voltages = nx.get_node_attributes(G, "v")

subax1 = plt.subplot(121)

nx.draw(G, with_labels=True, font_weight='bold')


# subax2 = plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')

plt.show() 

