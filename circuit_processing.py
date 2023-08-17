import cv2

import networkx as nx

import matplotlib.pyplot as plt

import random

import numpy as np

import copy

from ultralytics import YOLO

from text_extraction import extract_values


def get_closest_nodes(resistor_cords,nodes,vertical):
    
    print(nodes)
    
    resistor_x,resistor_y = resistor_cords
    
    tolerance = 50
    
    if vertical:
        
        resistor_nodes = filter(lambda x: resistor_x >= x[0] and resistor_x  <= x[2]  ,nodes)
        
        
        # deep copy the resistor nodes
        
        resistor_nodes1 = copy.deepcopy(resistor_nodes)
        resistor_nodes2 = copy.deepcopy(resistor_nodes)
        
        
        
        resistor_nodes_top = sorted(resistor_nodes1,key=lambda x: abs(resistor_y - x[1]))
        
        resistor_nodes_bottom = sorted(resistor_nodes2,key=lambda x: abs(x[3] - resistor_y))
        
        
        # get the closest node top and bottom of the resistor
        
        resistor_node1 = resistor_nodes_top[0]
        resistor_node2 = resistor_nodes_bottom[0]
        
        i = 1
        
        while resistor_node1 == resistor_node2:
            
            
            resistor_node2 = resistor_nodes_bottom[i]
            i += 1
            
        resistor_nodes = [resistor_node1,resistor_node2]

        
        
    else:
        
        resistor_nodes = filter(lambda x: resistor_y >= x[1] and resistor_y  <= x[3],nodes)
        
        
        
        # deep copy the resistor nodes
        
        resistor_nodes1 = copy.deepcopy(resistor_nodes)
        resistor_nodes2 = copy.deepcopy(resistor_nodes)
        
        
        
        
        resistor_nodes_right = list(sorted(list(resistor_nodes1),key=lambda x: abs(resistor_x - x[0])))
        
        resistor_nodes_left = list(sorted(list(resistor_nodes2),key=lambda x: abs(x[2] - resistor_x)))
        
        print(resistor_nodes_right,resistor_nodes_left)
        
        
        # get the closest node top and bottom of the resistor
        
        resistor_node1 = resistor_nodes_right[0]
        resistor_node2 = resistor_nodes_left[0]
        
        i = 1
        
        while resistor_node1 == resistor_node2:
            
            
            resistor_node2 = resistor_nodes_left[i]
            i += 1
            
            
            
        resistor_nodes = [resistor_node1,resistor_node2]
        
    
    
        
    return list(resistor_nodes)

def get_number_from_text(text):
    
    result = ""
    
    for c in text:
        
        if c.isdigit():
            
            result += c
            
    return int(result)    


def get_closest_text(resistor_box_cords,texts):
    
    x1,y1,x2,y2 = resistor_box_cords
    
    resistor_box_width = abs(x2 - x1)
    
    resistor_box_height = abs(y2 - y1)
    
    resistor_box_center = [x1 + resistor_box_width/2, y1 + resistor_box_height/2]
    
    resistor_box_center_x,resistor_box_center_y = resistor_box_center
    
    resistor_box_center_x = int(resistor_box_center_x)
    resistor_box_center_y = int(resistor_box_center_y)
    
    
    
    closest_text = sorted(texts,key=lambda x: abs(resistor_box_center_x - x['cords'][0]) + abs(resistor_box_center_y - x['cords'][1]))
    
    
    return list(closest_text)


def add_resistors_to_graph(G,r,nodes,texts):
    
    for resistor in r:
        
        resistor_box_cords = resistor['cords']
        
        x1,y1,x2,y2 = resistor_box_cords
        
        box_width = abs(x2 - x1)
        
        box_height = abs(y2 - y1)
        
        vertical = box_height > box_width
        
        resistor_cords = [x1 + box_width/2, y1 + box_height/2]
        
        
        # find the nodes that are closest to the resistor
        
        resistor_nodes = get_closest_nodes(resistor_cords,nodes,vertical)
        
        
        
        if len(resistor_nodes) < 2:
            continue
        
        resistor_node1 = resistor_nodes[-1]
        
        resistor_node2 = resistor_nodes[-2]
        
        resistor_node1 = "node" + str(resistor_node1[0]) + str(resistor_node1[1])
        
        resistor_node2 = "node" + str(resistor_node2[0]) + str(resistor_node2[1])
        
        
        resistor_text = get_closest_text(resistor_box_cords,texts)
        
        resistor_text = resistor_text[0]['text']
        
        
        
        # get number from text
        
        resistor_value = get_number_from_text(resistor_text)
        
        if 'k' in resistor_text or 'K' in resistor_text:
            resistor_value *= 1000
        
        if 'M' in resistor_text:
            resistor_value *= 1000000
            
        if 'm' in resistor_text:
            resistor_value *= 0.001
            
        if 'u' in resistor_text:
            resistor_value *= 0.000001
        
        if 'n' in resistor_text:
            resistor_value *= 0.000000001
            
        if 'p' in resistor_text:
            
            resistor_value *= 0.000000000001
        
        
        
        G.add_edge(resistor_node1,resistor_node2,r=resistor_value,cords=resistor_cords,box_cords=resistor_box_cords,start=resistor_node1,end=resistor_node2,vertical=vertical)
        

        
        

        
        
        
        
        
    
    
def extract_nodes(img,boxes):
    
    # delete all boxes
    
    for box in boxes:
        
        img = cv2.rectangle(img, (box['cords'][0],box['cords'][1]), (box['cords'][2],box['cords'][3]), (255,255,255), -1)
        
        
    



    # preprocess the image
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # Applying threshold
    threshold = cv2.threshold(blurred, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    
    

    # Initialize a new image to store
    # all the output components
    output = np.zeros(gray_img.shape, dtype="uint8")
    
    nodes = []

    text_offset = 5
    
    # Loop through each component
    for i in range(1, totalLabels):
        
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        
        if (area > 40):
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
            
            
            # add centroids
            center = (int(centroid[i][0]),int(centroid[i][1]))
            
            componentMask = cv2.circle(componentMask,center,5,(255,0,0), -1)
            
            # add node rects
            
            nodes.append([values[i, cv2.CC_STAT_LEFT]-20, values[i, cv2.CC_STAT_TOP]-20,values[i, cv2.CC_STAT_LEFT] + values[i, cv2.CC_STAT_WIDTH] + 20, values[i, cv2.CC_STAT_TOP] + values[i, cv2.CC_STAT_HEIGHT] + 20])
            
            # output = cv2.circle(output,center,5,(255,0,0), -1)
            
            output = cv2.putText(output,f"{values[i, cv2.CC_STAT_LEFT] - 20}, {values[i, cv2.CC_STAT_TOP] - 20},{values[i, cv2.CC_STAT_LEFT] + values[i, cv2.CC_STAT_WIDTH] + 20}, {values[i, cv2.CC_STAT_TOP] + values[i, cv2.CC_STAT_HEIGHT] + 20}",(values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP]-text_offset),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            
            output = cv2.rectangle(output, (values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP]), (values[i, cv2.CC_STAT_LEFT] + values[i, cv2.CC_STAT_WIDTH], values[i, cv2.CC_STAT_TOP] + values[i, cv2.CC_STAT_HEIGHT]), (255, 255, 255), -1)
            
            text_offset += 5
            
            # cv2.imshow(f"{i}", componentMask)
        


    cv2.imshow("Filtered Components", output)
   
    # cv2.waitKey(0)
    
    print(nodes)
    
    return nodes

        
        
def add_voltage_sources_to_graph(G,v,nodes):
    
    for voltage_source in v:
            
            voltage_source_box_cords = voltage_source['cords']
            
            x1,y1,x2,y2 = voltage_source_box_cords
            
            box_width = abs(x2 - x1)
            
            box_height = abs(y2 - y1)
            
            vertical = box_height > box_width
            
            voltage_source_cords = [x1 + box_width/2, y1 + box_height/2]
            
            
            # find the nodes that are closest to the resistor
            
            voltage_source_nodes = get_closest_nodes(voltage_source_cords,nodes,vertical)
            
           
            
            if len(voltage_source_nodes) < 2:
                continue
            
            voltage_source_node1 = voltage_source_nodes[0]
            
            voltage_source_node2 = voltage_source_nodes[1]
            
            voltage_source_node1 = "node" + str(voltage_source_node1[0]) + str(voltage_source_node1[1])
            
            voltage_source_node2 = "node" + str(voltage_source_node2[0]) + str(voltage_source_node2[1])
            
            G.add_edge(voltage_source_node1,voltage_source_node2,v=1,start=voltage_source_node1,end=voltage_source_node2,cords=voltage_source_cords,box_cords=voltage_source_box_cords,vertical=vertical)
            

    
        
        


def construct_graph(img,boxes,texts):
    
    
    # nodes = list(filter(lambda x: x['class'] == 'node', boxes))
    
    nodes = extract_nodes(img,boxes)
    
    r = list(filter(lambda x: x['class'] == 'r', boxes))
    
    v = list(filter(lambda x: x['class'] == 'v', boxes))
    
    G = nx.MultiGraph()
    
    # add the nodes to the graph
    
    G.add_nodes_from(list(map(lambda x: "node" + str(x[0]) + str(x[1]), nodes)))
    
    # add node coordinates to the graph
    
    for node in nodes:
            
        node_name = "node" + str(node[0]) + str(node[1])
        
        G.nodes[node_name]['cords'] = node

    
    # add resistors to the graph
    
    add_resistors_to_graph(G,r,nodes,texts)
    
    
    add_voltage_sources_to_graph(G,v,nodes)
    
    
    
    
    
    return G

def index_nodes(G):
    
    """
    adds index i to each node
    
    """
    
    for i,node in enumerate(G.nodes):
        
        G.nodes[node]['i'] = i

    
def process_circuit(image_name):

    # Load a model
    model = YOLO('bestest.pt')
        
        
        


    # load the image

    img = cv2.imread(image_name)

    # boxes = [{'class':'v','cords':[112,40,158,80]},{'class':'r','cords':[190,40,250,80]},{'class':'r','cords':[310,100,330,150]},
    #          {'class':'r','cords':[410,100,430,150]},{'class':'r','cords':[60,100,80,150]},{'class':'node','cords':[317,57,322,62]},
    #          {'class':'node','cords':[417,57,422,62]},{'class':'node','cords':[317,168,322,173]},{'class':'node','cords':[417,168,422,173]},
    #          {'class':'node','cords':[68,57,73,62]},{'class':'node','cords':[68,168,73,173]}]


    # boxes = [{'class':'v','cords':[112,40,158,80]},{'class':'r','cords':[190,40,250,80]},{'class':'r','cords':[310,100,330,150]},
    #         {'class':'r','cords':[410,100,430,150]},{'class':'r','cords':[60,100,80,150]}]

    # Run batched inference on a list of images
    results = model(image_name)  # return a list of Results objects
    results_data = results[0].boxes.data

    classes = results[0].names

    boxes = []

    for result in results_data:
        
        class_index = int(result[5])
        
        
        class_name = classes[class_index]
        
        if class_name == "Resistor":
            class_name = "r"
        
        if class_name == "DC_Source":
            class_name = "v"
        
        
        box = {'class':class_name,'cords':[int(result[0]),int(result[1]),int(result[2]),int(result[3])]}
        
        boxes.append(box)


        
    texts = extract_values(image_name)


    G = construct_graph(img.copy(),boxes,texts)
    
    index_nodes(G)
    
    

    subax1 = plt.subplot(121)

    nx.draw(G, with_labels=True, font_weight='bold')
        
    plt.show() 


    for box in boxes:
        
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        
        # draw a rectangle on the image with text above it
        
        img = cv2.putText(img, box['class'], (box['cords'][0],box['cords'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        img = cv2.rectangle(img, (box['cords'][0],box['cords'][1]), (box['cords'][2],box['cords'][3]), color, 2)

    # show the image

    cv2.imshow('image', img)

    cv2.waitKey(0)
    
    return G















