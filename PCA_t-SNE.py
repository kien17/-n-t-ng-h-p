import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def load_dataset(path, size=(64, 64)):
    X = []
    for file in os.listdir(path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(path, file))
            if img is None:
                continue
            img = cv2.resize(img, size)
            X.append(img.flatten())
    return np.array(X)



def load_labels_auto(mode):
    fname = f"labels_{mode}.json"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Không tìm thấy {fname}, không thể chạy AUTO mode.")
    with open(fname, "r") as f:
        print(f"AUTO-MODE: dùng labels từ {fname}")
        return json.load(f)



manual_labels = [
    # Dán labels vào đây nếu muốn chạy thủ công
    # Ví dụ: 0,1,2,0,1,2,...
]



def visualize(X, labels, title):
    X = np.array(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=10)
    plt.title(f"PCA ({title})")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, s=10)
    plt.title(f"t-SNE ({title})")
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng lệnh:")
        print("   py visualize.py kmeans")
        print("   py visualize.py dbscan")
        print("   py visualize.py hierarchical")
        print("   py visualize.py manual")
        sys.exit()

    mode = sys.argv[1].lower()

    # Load dataset
    X = load_dataset("data/input")

    # AUTO MODE
    if mode in ["kmeans", "dbscan", "hierarchical"]:
        labels = load_labels_auto(mode)
        visualize(X, labels, title=mode.upper())

    # MANUAL MODE
    elif mode == "manual":
        if len(manual_labels) == 0:
            print("MANUAL MODE nhưng manual_labels đang rỗng!")
            print("Hãy mở visualize.py và dán list labels của bạn vào manual_labels.")
            sys.exit()

        print("MANUAL-MODE: dùng labels từ biến manual_labels.")
        visualize(X, manual_labels, title="MANUAL")

    else:
        print("Sai chế độ. Chỉ được: kmeans | dbscan | hierarchical | manual")
