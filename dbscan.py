import os
import cv2
import numpy as np
import shutil
import math

def euclidean_distance(a, b):
    return math.sqrt(np.sum((a - b) ** 2))

def region_query(data, idx, eps):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def dbscan_custom(data, eps=10.0, min_samples=3):
    labels = [-1] * len(data)     # -1 = noise (nhiễu)
    visited = [False] * len(data)
    cluster_id = 0

    for i in range(len(data)):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = neighbors.copy()

            while queue:
                j = queue.pop(0)

                if not visited[j]:
                    visited[j] = True
                    neighbors_j = region_query(data, j, eps)

                    if len(neighbors_j) >= min_samples:
                        queue.extend(neighbors_j)

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

    return labels

def dbscan(input_dir, output_dir, eps=10.0, min_samples=3, resize=(64, 64)):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    images = []
    file_paths = []

    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img_resized = cv2.resize(img, resize)
                images.append(img_resized.flatten())
                file_paths.append(path)

    if not images:
        print("Không tìm thấy ảnh trong thư mục!")
        return

    images = np.array(images).astype(np.float32) / 255.0

    labels = dbscan_custom(images, eps=eps, min_samples=min_samples)

    import json
    with open("labels_dbscan.json", "w") as f:
        json.dump(labels, f)
    print("Saved labels_dbscan.json")

    
    unique_labels = set(labels)
    print("Các cụm tìm được:", unique_labels)

    for lbl in unique_labels:
        folder = "noise" if lbl == -1 else f"cluster_{lbl}"
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    for i, path in enumerate(file_paths):
        lbl = labels[i]
        folder = "noise" if lbl == -1 else f"cluster_{lbl}"
        shutil.copy(path, os.path.join(output_dir, folder, os.path.basename(path)))

if __name__ == "__main__":
    input_dir = "./data/input"
    output_dir = "./data/output_dbscan"

    dbscan(input_dir, output_dir, eps=10.0, min_samples=3)

# eps: 5.0 - 15.0
# min_samples: 3 - 5
