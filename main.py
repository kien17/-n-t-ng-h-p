from optics import Optics
from Preprocessing import Read_Dataset
import json
if __name__ == "__main__":
    X= Read_Dataset("data\input")
    optics= Optics(min_samples= 30, xi= 21)
    optics.fit(X)
    labels = optics.labels
    with open("labels_optics.json", "w") as f:
        json.dump(labels, f)
    print("Saved labels_optics.json")
    pass