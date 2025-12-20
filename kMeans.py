import os
import cv2
import numpy as np
import random
import shutil
import math
import time  # --- Thêm: dùng để đo thời gian ---

def sqr(x):
    return x * x

def distance(p1, p2):
    return math.sqrt(np.sum((p1 - p2) ** 2))

def min_position_cluster(p, clusters):
    min_dist = distance(p, clusters[0])
    pos = 0
    for i in range(1, len(clusters)):
        d = distance(p, clusters[i])
        if d < min_dist:
            min_dist = d
            pos = i
    return pos

def kmeans(input_dir, output_dir, K=3, resize=(64, 64), max_iters=1000000):
    # --- đo thời gian toàn bộ hàm kmeans ---
    start_total = time.time()

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

    images = np.array(images)

    clusters = [images[random.randint(0, len(images)-1)] for _ in range(K)]

    # --- bắt đầu đo thời gian cho phần phân cụm K-Means ---
    start_cluster = time.time()

    for it in range(max_iters):
        labels = []

        for p in images:
            pos = min_position_cluster(p, clusters)
            labels.append(pos)

        new_clusters = []
        for i in range(K):
            members = [images[j] for j in range(len(images)) if labels[j] == i]
            if len(members) > 0:
                new_clusters.append(np.mean(members, axis=0))
            else:
                new_clusters.append(clusters[i])

        new_clusters = np.array(new_clusters)
        if np.allclose(new_clusters, clusters):
            # print(f"K-Means dừng sau {it+1} vòng lặp")
            break
        clusters = new_clusters

    end_cluster = time.time()
    print(f"Thời gian chạy K-Means (phần phân cụm): {end_cluster - start_cluster:.4f} giây")

    for i, path in enumerate(file_paths):
        cluster_id = labels[i]
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(path, os.path.join(cluster_dir, os.path.basename(path)))

    import json
    with open("labels_kmeans.json", "w") as f:
        json.dump(labels, f)
    print("Saved labels_kmeans.json")

    end_total = time.time()
    print(f"Thời gian chạy K-Means (cả pipeline đọc ảnh + phân cụm + lưu): {end_total - start_total:.4f} giây")

if __name__ == "__main__":
    input_dir = "./data/input"
    output_dir = "./data/data_output/output_kmeans"
    K = 3

    kmeans(input_dir, output_dir, K)
