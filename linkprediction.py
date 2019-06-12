#12786442 Sazid Banna

import numpy as np
from sklearn.metrics import *
import networkx as nx
from networkit import linkprediction, nxadapter
import itertools as IT
import pandas as pd
import time

#ref: https://hackernoon.com/link-prediction-in-large-scale-networks-f836fcb05c88

def assign_label(pair, graph):
    u, v = pair[0], pair[1]
    return int(graph.has_edge(u, v)) #using networkx graph.has_edge because networkit graph.hasedge throws segfault
                                    #reason unknowns


def concatenate(node_set, label):
    dataset = pd.DataFrame({'nodes': node_set, 'label': label})
    return dataset


def main():
    start = time.time()
    train = pd.read_csv("CA-AstroPh-train.csv") #without using the processed CSV, the processed dataset has empty datarow error
    #train_sample = train.sample(n=1961) #low value doesn't throw segfault
    test = pd.read_csv("CA-AstroPh-test.csv")
    #test_sample = test.sample(n=1961)
    print("File Read:", time.time() - start)

    G = nx.from_pandas_edgelist(train, source='fromid', target='toid')
    F = nx.from_pandas_edgelist(test, source='fromid', target='toid')
    print(nx.info(G))
    print(nx.info(F))
    print("Print graph info: ", time.time() - start)

    train_graph = nxadapter.nx2nk(G)
    test_graph = nxadapter.nx2nk(F)
    training_set = linkprediction.MissingLinksFinder(train_graph).findAtDistance(2)

    testing_set = linkprediction.MissingLinksFinder(test_graph).findAtDistance(2)
    print("Missing link finder: ", time.time() - start)
    y_train = list(map(lambda x: assign_label(x, graph=F), training_set))
    print("labeling train graph: ", time.time() - start)
    y_test = list(map(lambda x: assign_label(x, graph=G), testing_set))
    print("labeling test graph: ", time.time() - start)
    train = concatenate(training_set, y_train)
    test = concatenate(testing_set, y_test)

    trainLPs = [
        linkprediction.CommonNeighborsIndex(train_graph), linkprediction.JaccardIndex(train_graph),
        linkprediction.AdamicAdarIndex(train_graph), linkprediction.ResourceAllocationIndex(train_graph),
        linkprediction.PreferentialAttachmentIndex(train_graph), linkprediction.AdjustedRandIndex(train_graph),
        linkprediction.NeighborhoodDistanceIndex(train_graph), linkprediction.TotalNeighborsIndex(train_graph),
        linkprediction.SameCommunityIndex(train_graph), linkprediction.UDegreeIndex(train_graph),
        linkprediction.VDegreeIndex(train_graph)
    ]

    testLPs = [
        linkprediction.CommonNeighborsIndex(test_graph), linkprediction.JaccardIndex(test_graph),
        linkprediction.AdamicAdarIndex(test_graph), linkprediction.ResourceAllocationIndex(test_graph),
        linkprediction.PreferentialAttachmentIndex(test_graph), linkprediction.AdjustedRandIndex(test_graph),
        linkprediction.NeighborhoodDistanceIndex(test_graph), linkprediction.TotalNeighborsIndex(test_graph),
        linkprediction.SameCommunityIndex(test_graph), linkprediction.UDegreeIndex(test_graph), linkprediction.VDegreeIndex(test_graph)
    ]

    X_train = linkprediction.getFeatures(training_set, *trainLPs)
    X_test = linkprediction.getFeatures(testing_set, *testLPs)

    features = ['CN', 'JC', 'AA', 'RA', 'PA', 'AR', 'ND', 'TN', 'SC', 'UD', 'VD']
    train_features = pd.DataFrame(X_train, columns=features)
    test_features = pd.DataFrame(X_test, columns=features)
    train = pd.concat([train, train_features], axis=1)
    test = pd.concat([test, test_features], axis=1)

    train.to_csv('train.csv', sep=';', header=True, decimal='.', encoding='utf-8', index=False)
    test.to_csv('test.csv', sep=';', header=True, decimal='.', encoding='utf-8', index=False)
    print("fin")
    print(time.time() - start)


main()


