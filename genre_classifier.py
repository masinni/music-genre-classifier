import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data_10.json"


def load_data(dataset_path):
    with open(dataset_path, 'r') as json_file:
        data = json.load(json_file)

    # convert lists into numpy arrayes
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)

    # split data intro train and test set
    X_train, X_test, y_train, y_test = train_test_split(inputs,
                                                        targets,
                                                        test_size=0.3,
                                                        random_state=42)

    # build the network archticture

    # compile network

    # train network
