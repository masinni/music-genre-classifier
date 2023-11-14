import os
import math
import librosa


DATASET_PATH = "Data/genres_original"
JSON_PATH = "data.json"
SAMPLERATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLERATE * DURATION


def save_mfcc(dataset_path,
              json_path,
              n_mfcc=13,
              n_fft=2048,
              hop_length=512,
              num_segments=5):

    # build dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],  # inputs
        "labels": []  # targets
    }

    num_samples_per_segmant = int(SAMPLES_PER_TRACK / num_segments)
    expectd_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segmant / hop_length)

    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")  # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]  # "blues"
            data["mapping"].append(semantic_label)

            # process files for a specific genre
            for f in filenames:

                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLERATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segmant * s  # s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segmant  # s=0 -> mun samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expectted length
                    if len(mfcc) == expectd_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)

