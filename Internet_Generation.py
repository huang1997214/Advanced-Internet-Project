import numpy as np
import random
import networkx


def Internet_Generation(node_num = 100, edge_gen_prob = 0.04):
    edge_list = []
    feature_list = []
    node_type_list = []
    rout_prob = 0.1
    #Generate the node type
    for i in range(node_num):
        probability = random.random()
        if probability<= rout_prob:
            node_type_list.append(0)
        else:
            node_type_list.append(1)
    #Generate the edge
    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                continue
            if [i, j] in edge_list:
                continue
            probability = random.random()
            if node_type_list[i] != node_type_list[j]:
                if probability <= 3 * edge_gen_prob:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
            else:
                if probability <= edge_gen_prob:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
    #Generate the node feature
    for i in range(node_num):
        temp_node_feat = [0, 0]
        for j in range(node_num):
            if [i, j] in edge_list:
                if node_type_list[i] != node_type_list[j]:
                    temp_node_feat[0] += 1
                else:
                    temp_node_feat[1] += 1
        feature_list.append(temp_node_feat)
    return feature_list, edge_list, node_type_list

def build_false_batch(node_num, edge_list, batch_size = 16):
    false_batch = []
    i = 0
    while i < batch_size:
        node_1 = random.randint(0, node_num - 1)
        node_2 = random.randint(0, node_num - 1)
        if [node_1, node_2] in edge_list:
            continue
        else:
            false_batch.append([node_1, node_2])
            i += 1
    return false_batch