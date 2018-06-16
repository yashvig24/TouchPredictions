import math
import numpy as np
import operator

def euclidian_distance(arr1, arr2):
    squared_diff = np.power((arr1 - arr2), 2)
    return math.sqrt(sum(squared_diff))

def k_nearest(entry, k, train, target):
    distances = []
    arr1 = np.array(entry.drop(target))
    for i in range(len(train)):
        arr2 = np.array(train.drop(target, axis = 1).iloc[i, :])
        dist = euclidian_distance(arr1, arr2)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors, classifier):
    if(classifier):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    else:
        resp = 0
        for x in neighbors:
            resp += x
        return resp/(1.0*len(neighbors))



