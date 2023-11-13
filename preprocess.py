import os


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # build dictionary to store data
    data = {
        "mapping": ["classical", "blues"],
        "mfcc": [[], [], []],  # inputs
        "labels": [0, 0, 1]  # targets
    }

    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        pass