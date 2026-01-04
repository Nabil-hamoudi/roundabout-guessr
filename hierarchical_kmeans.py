import numpy as np
from sklearn.cluster import KMeans
import torch

class Node:
    def __init__(self, depth, max_depth, k, center):
        self.depth = depth
        self.max_depth = max_depth
        self.k = k
        self.center = center
        self.ids = []
        self.children = []
        self.elems = []
    
class HKMeans:
    def __init__(self, elems, ids, max_depth, k, min_elems = 128):
        self.k = k
        self.max_depth = max_depth
        self.min_elems = min_elems

        self.root = self._split_node(elems, ids, 0) 

    def _split_node(self, elems, ids, depth):
        node = Node(depth, self.max_depth, self.k, np.mean(elems, 0))

        #noeud terminal ou pas
        #si terminal juste on met tt ce qu'on avait et gg
        if depth >= self.max_depth or len(elems) <= self.min_elems:
            node.ids = ids
            node.elems = elems
            return node


        #ATTENTIOn
        #Actuellement ça dit qu'il y a peu de clusters
        #On a soit un gros pb d'entraînement, soit un pb de dataset
        #Il faut regarder quand on aura un dataset
        # (et pas du bruit)
        kmeans = KMeans(self.k, n_init=10, random_state=67)
        labels = kmeans.fit_predict(elems)

        for i in range(self.k):
            #on récup un mask pour les elems du cluster
            mask = (labels == i)
            #si le masque est pas vide (sinon pb de liste vide)
            if np.any(mask):
                child_node = self._split_node(elems[mask], ids[mask], depth+1)
                node.children.append(child_node)
            
        return node

    def find_cluster(self, query):
        #parcours d'arbre classique
        cur = self.root

        if len(cur.children) == 0:
            return cur.ids, cur.elems

        while len(cur.children) > 0:

            closest = -1
            min_dist = float('inf')

            for elem in cur.children:
                d = np.linalg.norm(query - elem.center)
                if d < min_dist:
                    min_dist = d
                    closest = elem
            if closest == -1: break
            cur = closest
        
        return closest.ids, closest.elems

    def find_elem(self, query, top_k=5):
        """
        Trouve les Top-K éléments les plus proches dans le cluster identifié.
        """

        real_k = top_k

        ids, elems = self.find_cluster(query)
        distances = np.linalg.norm(elems - query, axis=1)
        k = min(real_k, len(ids))
        top_indices = np.argpartition(distances, k-1)[:k]
        top_indices = top_indices[np.argsort(distances[top_indices])]
        vu = set()
        print(top_indices)
        results = []
        for idx in top_indices:
            if not ids[idx] in vu:
                results.append({
                    "id": ids[idx],
                    "distance": distances[idx]
                })
            
        return results


