import os
import math
import librosa
import json

DATASET_PATH = "genre_dataset_ruduced"
JSON_PATH = "data_10.json"
SAMPLERATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLERATE * DURATION


def save_mfcc(dataset_path,
              json_path,
              n_mfcc=13,
              n_fft=2048,
              hop_length=512,
              num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file
       along witgh genre labels.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :param: num_segments (int): Number of segments we want to divide sample tracks into
    :return:
    """

    # build dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],  # inputs
        "labels": []  # targets
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expectd_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")  # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]  # "blues"
            data["mapping"].append(semantic_label)
            print(f"\nProcessing {semantic_label}")

            # process files for a specific genre
            for f in filenames:

                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLERATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start = num_samples_per_segment * s  # s=0 -> 0
                    finish = start + num_samples_per_segment  # s=0 -> num samples_per_segment

                    # extract MFCC features for the segment
                    mfcc = librosa.feature.mfcc(y=signal[start:finish],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expectted length
                    if len(mfcc) == expectd_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print(f"{file_path}, segment:{s+1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)