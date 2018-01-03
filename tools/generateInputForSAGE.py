#!/usr/bin/python

import sys, os
import json
import numpy as np
import networkx as nx
import random
import argparse

from networkx.readwrite import json_graph
from argparse import ArgumentParser
from copy import deepcopy

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

def getRandomFeatures(feature_size):
    fea = [random.random() for i in range(feature_size)]
    #fea = []
    #for i in range(feature_size):
     #   x = random.random()
     #   if x < 0.02:
     #       fea.append(1.0)
      #  else:
       #     fea.append(0.0)
    return fea

def generateInputForSAGE(initFile, dynamicFile, flagFile, dataname, outputPath):
    Gpath = str(outputPath)+"/"+str(dataname)+"-G.json"
    id_map_path = str(outputPath)+"/"+str(dataname)+"-id_map.json"
    class_map_path=str(outputPath)+"/"+str(dataname)+"-class_map.json"
    feats_path = str(outputPath)+"/"+str(dataname)+"-feats.npy"
    # read node flag
    node_flag = {}
    flag_set = []
    flag_no = 0
    with open(flagFile, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            items = line.split()
            if len(items) < 2:
                continue
            node_flag[int(items[0])] = int(items[1])
            if int(items[1]) not in flag_set:
                flag_set.append(int(items[1]))
                flag_no += 1

    label_vec_ori = [0 for i in range(flag_no)]
    
    edges_init = readEdgesFile(initFile)
    edges_dyn = readEdgesFile(dynamicFile)
    
    class_map = {}
    id_map = {}
    # generate networkx
    G = nx.Graph(name="disjoint_union(, )")
    feature_size = 10
    feats = []
    # add train node
    for it in edges_init:
        sn = it[0]
        tn = it[1]
        if sn not in G:
            fea_lst = getRandomFeatures(feature_size)
            label_lst = deepcopy(label_vec_ori)
            label_lst[node_flag[sn]-1]=1
            G.add_node(sn, test=False, feature=fea_lst, val=False, label=label_lst)
            class_map[str(sn)] = label_lst
            id_map[str(sn)] = sn
            feats.append(fea_lst)
        if tn not in G:
            fea_lst = getRandomFeatures(feature_size)
            label_lst = deepcopy(label_vec_ori)
            label_lst[node_flag[tn]-1]=1
            G.add_node(tn, test=False, feature=fea_lst, val=False, label=label_lst)
            class_map[str(tn)] = label_lst
            id_map[str(tn)] = tn
            feats.append(fea_lst)

        G.add_edge(sn, tn, test_removed=False, train_removed=False)

    # add val and test node
    for it in edges_dyn:
        sn = it[0]
        tn = it[1]
        if sn not in G:
            fea_lst = getRandomFeatures(feature_size)
            label_lst = deepcopy(label_vec_ori)
            label_lst[node_flag[sn]-1]=1
            G.add_node(sn, test=True, feature=fea_lst, val=True, label=label_lst)
            class_map[str(sn)] = label_lst
            id_map[str(sn)] = sn
            feats.append(fea_lst)

        if tn not in G:
            fea_lst = getRandomFeatures(feature_size)
            label_lst = deepcopy(label_vec_ori)
            label_lst[node_flag[tn]-1]=1
            G.add_node(tn, test=True, feature=fea_lst, val=True,label=label_lst)
            class_map[str(tn)] = label_lst
            id_map[str(tn)] = tn
            feats.append(fea_lst)

        G.add_edge(sn, tn, test_removed=True, train_removed=True)

    with open(Gpath, 'w') as outfile:
        outfile.write(json.dumps(json_graph.node_link_data(G)))

    with open(id_map_path, 'w') as outfile:
        outfile.write(json.dumps(id_map))

    with open(class_map_path, 'w') as outfile:
        outfile.write(json.dumps(class_map))
    np.save(feats_path, feats)

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--initFile', type=str, help="the init graph from DNE")
    parser.add_argument('--dynamicFile', type=str, help="the dynamic graph from DNE")
    parser.add_argument('--flagFile', type=str, help="the flag file from DNE")
    parser.add_argument('--dataname', type=str, help="the output file name")
    parser.add_argument('--outputPath', type=str,help="output path")
    args = parser.parse_args()
    initFile = args.initFile
    dynamicFile = args.dynamicFile
    flagFile = args.flagFile
    dataname = args.dataname
    outputPath = args.outputPath

    generateInputForSAGE(initFile, dynamicFile, flagFile, dataname, outputPath)
