import numpy as np
import sys
import json, csv
from scipy.stats import spearmanr
import math
from sklearn import cluster
from sklearn import metrics

class Embedding():
    def __init__(self,filename):
        self.model = self.load_embeddings(filename)

    def cosine_similarity(self,v1,v2):
      "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
      sumxx, sumxy, sumyy = 0, 0, 0
      for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
      return sumxy/math.sqrt(sumxx*sumyy)

    def sorted_by_similarity(self,word):
        """Returns words sorted by cosine distance to a given vector, most similar first"""
        model = self.model
        if word in model:
            base_vector = model[word]
        else:
            return None
        words_with_distance = [(self.cosine_similarity(base_vector, model[w]),w) for w in model][0:10]
        # We want cosine similarity to be as large as possible (close to 1)
        return sorted(words_with_distance, key=lambda t: t[0], reverse=True)


    def load_embeddings(self,filename):
        f = open(filename,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model

    def clusters(self,N):
        y = list(self.model.keys())
        X = list(self.model.values())[1:100]
        kmeans = cluster.KMeans(n_clusters=N)
        kmeans.fit(X)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    
        centroids_closest = {}
        for index,x in enumerate(X):
            label = labels[index]
            centroid = centroids[label]
            dist = self.cosine_similarity(x,centroid)
            if label in centroids_closest:
                if centroids_closest[label]['dist'] > dist:
                    centroids_closest[label]['dist'] = dist
                    centroids_closest[label]['index'] = index
                    centroids_closest[label]['points'].append(index)
            else:
                centroids_closest[label] = {}
                centroids_closest[label]['dist'] = dist
                centroids_closest[label]['index'] = index
                centroids_closest[label]['points'] = [index]

        print ("Cluster id labels for inputted data")
        print (labels)
        print ("Centroids data")
        for key,value in centroids_closest.items():
            print(key,y[value['index']],[y[index] for index in value['points'][0:5]])
        #print (centroids)


if __name__ == "__main__":
    #word = sys.argv[1]
    model = Embedding("embedding.txt")
    #print(model.sorted_by_similarity(word))
    N = 5
    model.clusters(N)
