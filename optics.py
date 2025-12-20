import numpy as np
import queue
import time  # --- Thêm: dùng để đo thời gian ---

class Optics:
    def __init__(self, eps=float('inf'), min_samples=5, xi=0.05):
        self.eps = eps
        self.min_samples = min_samples
        self.processed = None
        self.core_distances = None
        self.ordering = []
        self.reachability_distances = None
        self.xi = xi
        self.labels = None
        self.n = 0

    def fit(self, X):
        # --- đo thời gian cho toàn bộ quá trình OPTICS.fit ---
        start_time = time.time()

        n = len(X)
        self.n = n
        self.processed = [False] * n
        self.core_distances = [float('inf')] * n
        self.reachability_distances = [float('inf')] * n
        self.ordering = []

        for i in range(len(X)):
            if not self.processed[i]:
                self.expand_cluster(i, X)

        self.extract_cluster(self.xi)

        end_time = time.time()
        print(f"Thời gian chạy OPTICS (fit): {end_time - start_time:.4f} giây")

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def find_neighbors(self, point_idx, X):
        # Placeholder for neighbor finding logic
        neighbors = []
        for i in range(len(X)):
            if i != point_idx:
                distant: float = self.euclidean_distance(X[point_idx], X[i])
                if distant <= self.eps:
                    neighbors.append(i)
        return neighbors

    def expand_cluster(self, point_idx, X):
        # Placeholder for cluster expansion logic
        neighbors = self.find_neighbors(point_idx, X)
        self.processed[point_idx] = True
        self.ordering.append(point_idx)
        if len(neighbors) < self.min_samples:
            return

        distances = [self.euclidean_distance(X[point_idx], X[j]) for j in neighbors]
        distances.sort()
        self.core_distances[point_idx] = distances[self.min_samples - 1]

        # Initialize priority queue for reachability distances
        pqueue = queue.PriorityQueue()
        for neighbor in neighbors:
            if not self.processed[neighbor]:
                distant = self.euclidean_distance(X[point_idx], X[neighbor])
                new_reach_dist = max(self.core_distances[point_idx], distant)
                if new_reach_dist < self.reachability_distances[neighbor]:
                    self.reachability_distances[neighbor] = new_reach_dist
                    pqueue.put((new_reach_dist, neighbor))

        while not pqueue.empty():
            _, q = pqueue.get()
            if not self.processed[q]:
                self.expand_cluster(q, X)

    def extract_cluster(self, eps_cluster):
        self.labels = [-1] * self.n
        current_label = -1

        for idx in self.ordering:
            if self.reachability_distances[idx] > eps_cluster:
                # Điểm này xa → bắt đầu cụm mới
                current_label += 1
                self.labels[idx] = current_label
            else:
                # điểm này thuộc cụm hiện tại
                self.labels[idx] = current_label

        return self.labels


# --- Ví dụ sử dụng & đo thời gian toàn pipeline (nếu bạn có dữ liệu ảnh giống 2 file kia) ---
if __name__ == "__main__":
    import os
    import cv2

    input_dir = "./Dataset"

    images = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                images.append(img_resized.flatten())

    if not images:
        print("Không tìm thấy ảnh trong thư mục!")
    else:
        X = np.array(images).astype(np.float32) / 255.0

        start_total = time.time()
        optics = Optics(eps=float("inf"), min_samples=30, xi=0.05)
        optics.fit(X)
        end_total = time.time()

        print(f"Thời gian chạy OPTICS (cả pipeline đọc ảnh + fit): {end_total - start_total:.4f} giây")
