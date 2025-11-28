import numpy  as np
import math
import cv2
import os
import shutil
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def load_data(path):
    X= []
    file_path= []
    for file in os.listdir(path):
        img_path= os.path.join(path, file)
        img= cv2.imread(img_path)
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize= cv2.resize(gray, (64,64))
        X.append(resize.astype(np.uint8))
        file_path.append(img_path)
    return np.array(X), file_path

class MODEL:
    def __init__(self, children, distance, labels):
        self.children_= children
        self.distances_= distance
        self.labels_= labels

def euclidean_distance(x, y):
    return np.linalg.norm((x-y),'fro')

def cluster_distance(idx_A,idx_B, X, children):
    n= len(X)
    A=[idx_A]
    B=[idx_B]
    while max(A) >= n:
        new_A= []
        for child in range(0, len(A)):
            if A[child] >= n:
                new_A.extend(children[A[child] - n])
            else:
                new_A.append(A[child])
        A= new_A
    
    while max(B) >= n:
        new_B= []
        for child in range(0, len(B)):
            if B[child] >= n:
                new_B.extend(children[B[child] - n])
            else:
                new_B.append(B[child])
        B= new_B
    min_distance= -1
    for child1 in A:
        for child2 in B:
            if min_distance == -1 or min_distance > euclidean_distance(X[child1], X[child2]):
                min_distance= euclidean_distance(X[child1], X[child2])
    return min_distance

def AgglomerativeClustering(X):
    n= len(X)
    max_cluster_size= 2 * n - 2
    if (max_cluster_size <= 0):
        print("Error Input\n")
        return
    children= []
    distance= []
    labels= [i for i in range(0, n)]
    visited= [0 for _ in range(0, max_cluster_size + 1)]
    number_of_clusters= n-1
    distance_matrix= [[0 for _ in range(0, max_cluster_size + 1)] for _ in range(0, max_cluster_size + 1)] 
    for i in range(0, n):
        for j in range(0, n):
            distance_matrix[i][j]= euclidean_distance(X[i], X[j])
    
    cluster_idx= n
    while(number_of_clusters > 1):
        child1= -1
        child2= -1
        child_distance= 0

        for i in range(0, max_cluster_size + 1):
            for j in range(0, max_cluster_size + 1):
                if distance_matrix[i][j] != 0 and (child_distance == 0 or child_distance > distance_matrix[i][j]):
                    child1= i
                    child2= j
                    child_distance= distance_matrix[i][j]

        children.append([child1, child2])
        distance.append(distance_matrix[child1][child2])
        distance_matrix[child1][child2]= 0
        distance_matrix[child2][child1]= 0

        m= len(children) - 1
        point_idx= children[m]
        while max(point_idx) >= n:
            new_point_idx= []
            for i in range(0, len(point_idx)):
                if point_idx[i] >= n:
                    new_point_idx.extend(children[point_idx[i] - n])
                else:
                    new_point_idx.append(point_idx[i])
            point_idx= new_point_idx
        
        for i in point_idx:
            labels[i]= m

        visited[child1]= 1
        visited[child2]= 1
        for i in range(0, cluster_idx):
            distance_matrix[i][child1]= 0
            distance_matrix[child1][i]= 0

            distance_matrix[child2][i]= 0
            distance_matrix[i][child2]= 0
            if not visited[i] and i < len(distance_matrix) and cluster_idx < len(distance_matrix[0]):
                distance_matrix[i][cluster_idx]= cluster_distance(i, cluster_idx, X, children)

        cluster_idx+= 1
        number_of_clusters= len(list(set(labels)))

    return children, distance, labels

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("distance_matrix")
    plt.show()

def Saving(model, file_path):
    n = len(model.labels_)
    folder_path= "cluster"
    st= []
    st.append(folder_path)
    cluster= []
    cluster.append(len(model.children_) - 1 + n)
    while len(st) != 0:
        i= cluster.pop()
        merge= model.children_[i - n]
        folder_path= st.pop()
        os.makedirs(folder_path, exist_ok=True)
        for child in merge:
            if child < n:
                 dst= os.path.join(folder_path, f"{child}.jpg")
                 shutil.copy2(file_path[child], dst)
            else:
                new_folder= os.path.join(folder_path, f"cluster{child}")
                st.append(new_folder)
                cluster.append(child)

    all_picture_path= "order_picture"
    os.makedirs(all_picture_path, exist_ok= True)

    for i in range(0, len(file_path)):
        dst= os.path.join(all_picture_path, f"{i}.jpg")
        shutil.copy2(file_path[i], dst)

    


def main():
    X_loaded, file_path= load_data("picture")
    children, distance, labels= AgglomerativeClustering(X_loaded)
    agg= MODEL(np.array(children), np.array(distance), np.array(labels))
    plot_dendrogram(agg, truncate_mode=None)
    Saving(agg, file_path)

if __name__ == "__main__":
    main()