import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


from circuit_processing import process_circuit

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
    
    
    
    
    node_indices = nx.get_node_attributes(G, "i")
    
    
    known_voltage_nodes = set()
    
    
    for i, node in enumerate(G.nodes):
        
        
    
        
        if i >= n:
            break
        
        
        
        
        edge_nodes = list(set(G.edges(node)))
        
        
        
        for edge_node in edge_nodes:
        
            edges = G.get_edge_data(edge_node[0],edge_node[1])
            
            # switch nodes if edge is directed from gnd to node
            
            if edge_node[0] == gnd:
                edge_node = (edge_node[1],edge_node[0])
            
            
            
            for j,attr in edges.items():
                
                node1_idx = node_indices[edge_node[0]]
                
            
                if 'v' in attr and edge_node[1] == gnd:
                    
                    nx.set_node_attributes(G,{attr['start']:{'v':attr['v']}})
                    known_voltage_nodes.add(attr['start'])
          
    return len(known_voltage_nodes)
    
__node_edges_keys_set = {}    
    
def merge_nodes(G, n1, n2,voltage):
    
    supernode_name = f"{n1}_{n2}"
    
    node_indices = nx.get_node_attributes(G, "i")
    
    node1_idx = node_indices[n1] if n1 in node_indices else original_node_indices[n1]  
    node2_idx = node_indices[n2] if n2 in node_indices else original_node_indices[n2] 
    
    G.add_node(supernode_name,supernode=True,v=voltage,node1_index=node1_idx,node2_index=node2_idx,node1=n1,node2=n2)
    node1_edges = G.edges(n1,data=True)
    
    node2_edges = G.edges(n2,data=True)
    
    
    
    for node1_edge in node1_edges:
        
        if node1_edge[1] == n2:
            continue
        
        G.add_edge(supernode_name,node1_edge[1])
        
        edge_attrs = node1_edge[2]
        
        edge_attrs['node'] = n1
        
        if (supernode_name,node1_edge[1]) in __node_edges_keys_set:
            
            
            
            key = __node_edges_keys_set[(supernode_name,node1_edge[1])] + 1
            
           
            
        else:
            key = 0
            
        __node_edges_keys_set[(supernode_name,node1_edge[1])] = key
            
       
        
        nx.set_edge_attributes(G, {(supernode_name,node1_edge[1],key):edge_attrs})
        
        
        
    G.remove_node(n1)
    
    
    
    for node2_edge in node2_edges:
        if node2_edge[1] == n1:
            continue
        
        G.add_edge(supernode_name,node2_edge[1])
        
        edge_attrs = node2_edge[2]
        
        edge_attrs['node'] = n2
        
        if (supernode_name,node2_edge[1]) in __node_edges_keys_set:
            key = __node_edges_keys_set[(supernode_name,node2_edge[1])] + 1
            
           
            
        else:
            key = 0
       
        __node_edges_keys_set[(supernode_name,node2_edge[1])] = key
        
        nx.set_edge_attributes(G, {(supernode_name,node2_edge[1],key):edge_attrs})
        
        
    
    
    G.remove_node(n2)
    
    # print(G.edges(supernode_name,data=True))

supernodes_nodes_indices = []

    
def adjust_for_supernodes(G,gnd,A,voltages,row):
    
    
    
    battery_edges = nx.get_edge_attributes(G,'v')
    
    for battery_edge,voltage in battery_edges.items():
        node1,node2,_ = battery_edge
        
        if node1 == gnd or node2 == gnd:
            continue
        
        
        
        node1_idx = G.nodes[node1]['i'] if node1 in G.nodes else original_node_indices[node1]
        
        node2_idx = G.nodes[node2]['i'] if node2 in G.nodes else original_node_indices[node2]
        
        supernodes_nodes_indices.append(node1_idx)
        supernodes_nodes_indices.append(node2_idx)
        
        
        
        A[row][node1_idx] += 1
        
    
       
        A[row][node2_idx] -= 1
        
        voltages[row] += voltage
        
        row -= 1
        
        
        
        
        merge_nodes(G,node1,node2,voltage)
        
        
        
        

    
def compute_normal_node_edges(G,row,edges,A,voltages,node_name,node_attrs,gnd):
    
    
    for edge in edges:
        
        
        
        node1, node2, edge_attrs = edge
        
        supernode2 = 'supernode' in G.nodes[node2]
        
        
        
        if(supernode2):
            
            node2_idx = G.nodes[node2]['node1_index'] if G.nodes[node2]['node1'] == edge_attrs['node'] else G.nodes[node2]['node2_index']
            
            node2 = edge_attrs['node']
            
        else:
            node2_idx = G.nodes[node2]['i']
            
        
            
        
            
        
        
        node1_idx = node_attrs['i']
        
       
        
        
        if 'i' in edge_attrs:
                
                if node1 == edge_attrs['start']:
                
                    voltages[row] += edge_attrs['i']
                else:
                    voltages[row] -= edge_attrs['i']
                
                continue
        
        
                
        if 'v' in edge_attrs and 'r' in edge_attrs:
            
            A[row][node1_idx] += 1 / edge_attrs['r']
            
            if node1 == edge_attrs['start']:
            
                voltages[row] += edge_attrs['v'] / edge_attrs['r']
            else:
                voltages[row] -= edge_attrs['v'] / edge_attrs['r']
            
            
            
        elif 'r' in edge_attrs:
            
            A[row][node1_idx] += 1 / edge_attrs['r']
            
            
            
            if supernode2 or (node2 != gnd and 'v' not in G.nodes[node2]):
                A[row][node2_idx] -= 1 / edge_attrs['r']
                
                
               
            elif 'v' in G.nodes[node2]:
                voltages[row] += G.nodes[node2]['v'] / edge_attrs['r']
                
    
    
        
        
def compute_supernode_edges(G,row, edges,A,voltages,node_name,node_attrs,gnd ):
    
    supernode_node1 = node_attrs['node1']
    
    supernode_node2 = node_attrs['node2']
    
    supernode_node1_idx = node_attrs['node1_index']
    
    supernode_node2_idx = node_attrs['node2_index']
    
    supernode_voltage = node_attrs['v']
    
    
    

    
    for edge in edges:
        
        node1, node2, edge_attrs = edge
        
        
        
        node2_idx = G.nodes[node2]['i']
        
        # print(node1, node2,edge_attrs)
        
        idx = supernode_node1_idx if supernode_node1 == edge_attrs['node'] else supernode_node2_idx
        
        A[row][idx] += 1 / edge_attrs['r']
        
        if 'v' in G.nodes[node2]:
            
            voltages[row] += G.nodes[node2]['v'] / edge_attrs['r']
       
        
        if node2 != gnd and 'v' not in G.nodes[node2]:
            
            A[row][node2_idx] -= 1 / edge_attrs['r']
            
            # print(node2,node2_idx,edge_attrs['r'],row)
        
        
        

    
    
def reindex_graph(G, gnd):
    
    
    i = 0
    
   
    
    for node in G.nodes(data=True):
        
        if node[0] == gnd or 'supernode' in node[1] or 'v' in node[1]:
            
            continue
        
        while i in supernodes_nodes_indices:
            i += 1
            
        
        nx.set_node_attributes(G,{node[0]:{'i':i}})
        
        i += 1
        
        
        
    
    # print(G.nodes(data=True))

def store_original_node_indices(G):
    
    for node in G.nodes:
        
        
        
        original_node_indices[node] = G.nodes[node]['i']


def construct_nodal_equations(G,gnd):
    
    A_row = 0
    
    
    
    
    
    known_voltages_num = add_known_voltages_to_nodes(G,gnd)
    
    
    
    n = len(G.nodes) - 1 - known_voltages_num
    
    print(f'Total nodes: {len(G.nodes)}, Known voltages: {known_voltages_num}, Unknown voltages: {n}\n\n')
    
    
    A = np.zeros((n,n))
    
    voltages = np.zeros((n,1))
    
    reindex_graph(G,gnd)
    
    store_original_node_indices(G)
    
    adjust_for_supernodes(G,gnd,A,voltages,-1)
    
    reindex_graph(G,gnd)
    
    
    
    nodes = list(G.nodes(data=True))
    
    # print(nodes)
    
    for node in nodes:
        node_name, node_attrs = node
        
       
        
        if node_name == gnd:
            continue
        
        edges = G.edges(node_name,data=True)
        
        
        
        if 'supernode' in node_attrs:
            compute_supernode_edges(G,A_row,edges,A,voltages,node_name,node_attrs,gnd)
        
            continue
        
        
        
        if 'v' not in node_attrs:
        
            compute_normal_node_edges(G,A_row,edges,A,voltages,node_name,node_attrs,gnd)
            
            A_row += 1

    print(A,voltages)
    
    
    return np.linalg.solve(A,voltages)
    
    
def add_calculated_voltages_to_nodes(G,node_voltages,gnd):
    
    
    # sort nodes based on index

    
    nodes = sorted(G.nodes(data=True),key=lambda node: node[1]['i'])
    
    for node in nodes:
       
        
        node_name, node_attrs = node
        
        if 'v' in node_attrs:
            continue
        
        if node_name == gnd:
            # set v = 0 for gnd node
            
            nx.set_node_attributes(G,{node_name:{'v':0}})
            
            continue
        
        G.nodes[node_name]['v'] = node_voltages[node_attrs['i']][0]
    # print(G.nodes(data=True))
        
   
def add_currents_to_edges(G,gnd):
        
    edges = G.edges(data=True)
    
    
    
    for key,edge in enumerate(edges):
        
        node1, node2, edge_attrs = edge
        
        if 'v' in edge_attrs or 'i' in edge_attrs:
           continue
        
        # switch nodes if edge is directed from gnd to node
        
        if node1 == gnd:
            node1, node2 = node2, node1
            
        
        node1_voltage = G.nodes[node1]['v']
        
        node2_voltage = G.nodes[node2]['v']
        
        
        
        edge_attrs['i'] = (node1_voltage - node2_voltage) / edge_attrs['r']
        
        nx.set_edge_attributes(G, {(node1,node2,key):edge_attrs})
            
    
def get_ground(G):
    """
    Getting ground by getting node with most connections
    
    """
    
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)

    
    
    return degrees[0][0]

 
  

# G = nx.MultiGraph()


# G.add_node('v1',i = 2)

# G.add_node('v2', i = 3)

# G.add_node('v3', i = 0)

# G.add_node('v4', i = 1)





# G.add_edge('v1','v2',v=100,start='v1',end='v2')

# G.add_edge('v1','v3',r=10)

# G.add_edge('v1','v3',r=5)

# G.add_edge('v3','v4',v=50,start='v3',end='v4')

# G.add_edge('v4','v2',r=15)

# G.add_edge('v4','v2',r=20)


def adjust_indices(G,G_copy):
    
    for node in G_copy.nodes(data=True):
        
        node_name, node_attrs = node
        
        if 'supernode' in node_attrs:
            
            node1 = node_attrs['node1']
            
            node2 = node_attrs['node2']
            
            node1_idx = G.nodes[node_attrs['node1']]['i']
            
            node2_idx = G.nodes[node_attrs['node2']]['i']
            
            G.nodes[node1]['i'] = node1_idx
            
            G.nodes[node2]['i'] = node2_idx
            
            continue
        
        
        G.nodes[node_name]['i'] = G_copy.nodes[node_name]['i']
        

def add_results_to_image(G,image_name,result_image_name):
    
    image = cv2.imread(image_name)
    
    result_image = image.copy()
    
    current_color = (0,0,255)
    
    voltage_color = (0,255,0)
    
    padding = 0
    
    nodes = set()
    
    # add padding to image
    
    # result_image = cv2.copyMakeBorder(result_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT,value=(255,255,255))
    
    for edge in G.edges(data=True):
        
        node1, node2, edge_attrs = edge
        
        if 'i' in edge_attrs:
            
            text = f"{edge_attrs['i']:.2f} A"
            
            node1_pos = G.nodes[node1]['cords']
            
            node2_pos = G.nodes[node2]['cords']
            
            x,y = edge_attrs['cords']
            
            box_cords = edge_attrs['box_cords']
            
            x = int(x) + padding 
            y = int(y) + padding
            
           
            
            
            
            # draw current arrow
            
            if edge_attrs['vertical']:
                cv2.arrowedLine(result_image, (box_cords[2] + padding,box_cords[1] + padding), (box_cords[2] + padding,box_cords[3] + padding), current_color, 2)
                cv2.putText(result_image, text,  (box_cords[2] + padding + 10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
            else:
                cv2.arrowedLine(result_image, (box_cords[0] + padding,box_cords[1] + padding), (box_cords[2] + padding,box_cords[1] + padding), current_color, 2)
                cv2.putText(result_image, text,  (x - padding // 2,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
            
        
        # add voltages to nodes
        
        if 'v' in G.nodes[node1] and node1 not in nodes:
            
            text = f"{G.nodes[node1]['v']:.2f} V"
            
            box_cords = edge_attrs['box_cords']
            
            x,y = edge_attrs['cords']
            
            x = int(x) + padding
            
            y = int(y) + padding
            
            if edge_attrs['vertical']:
                
                cv2.circle(result_image, (x + padding,box_cords[1] + padding), 5, voltage_color, -1)
                
                cv2.putText(result_image, text,  (x + padding - 20, box_cords[1] + padding - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voltage_color, 2)
                
            else:
                
                cv2.circle(result_image, (box_cords[2] + padding,y + padding), 5, voltage_color, -1)
                
                cv2.putText(result_image, text,  (box_cords[2] + padding,y + padding + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voltage_color, 2)
                
            nodes.add(node1)
            
        if 'v' in G.nodes[node2] and node2 not in nodes:
            
            text = f"{G.nodes[node2]['v']:.2f} V"
            
            box_cords = edge_attrs['box_cords']
            
            x = int(x) + padding
            
            y = int(y) + padding
            
            if edge_attrs['vertical']:
                
                cv2.circle(result_image, (x + padding,box_cords[3] + padding), 5, voltage_color, -1)
                
                cv2.putText(result_image, text,  (x + padding - 20, box_cords[3] + padding - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voltage_color, 2)
                
            else:
                
                cv2.circle(result_image, (box_cords[0] + padding,y + padding), 5, voltage_color, -1)
                
                cv2.putText(result_image, text,  (box_cords[0] + padding,y + padding + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voltage_color, 2)
            
            nodes.add(node2)
            
            
    for node in G.nodes(data=True):
        
        node_name, node_attrs = node
        
        if 'v' in node_attrs and node_name not in nodes:
            
            text = f"{node_attrs['v']:.2f} V"
            
            box_cords = node_attrs['box_cords']
            
            x,y = node_attrs['cords']
            
            x = int(x) + padding
            
            y = int(y) + padding
            
            cv2.circle(result_image, (x + padding,y + padding), 5, voltage_color, -1)
                
            cv2.putText(result_image, text,  (x + padding,y + padding - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voltage_color, 2)
            
            nodes.add(node_name)
    
    # cv2.imwrite(result_image_name,result_image)
            
    cv2.imshow("result",result_image)
    
    cv2.waitKey(0)
            

original_node_indices = {}      


def main():

    
    image_name = "circuit.png"


    G = process_circuit(image_name)

    G_copy = G.copy()
    
    


    # T = nx.minimum_spanning_tree(G)

    # print(G.get_edge_data('v1','v2'))

    gnd = get_ground(G_copy)


    print(f'gnd: {gnd}\n\n')
    

    node_voltages = construct_nodal_equations(G_copy,gnd)
    
    
    
    adjust_indices(G,G_copy)

    

    add_calculated_voltages_to_nodes(G,node_voltages,gnd)

    add_currents_to_edges(G,gnd)

    # node_voltages = nx.get_node_attributes(G, "v")

    # print(gnd)

 
    
    add_results_to_image(G,image_name,"circuit_result.png")

    # print(node_voltages)

    subax1 = plt.subplot(121)

    nx.draw(G_copy, with_labels=True, font_weight='bold')


    # subax2 = plt.subplot(122)
    # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')

    plt.show() 
    
    
main()



