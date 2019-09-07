import numpy as np
import sys
import json, csv
from scipy.stats import spearmanr
import math

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
        words_with_distance = [(self.cosine_similarity(base_vector, model[word]),w) for w in model]
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


if __name__ == "__main__":
    word = sys.argv[1]
    model = Embedding("embedding.txt")
    print(model.sorted_by_similarity(word))
