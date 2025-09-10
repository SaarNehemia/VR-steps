import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from matplotlib import use
use('TkAgg')

def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


if __name__ == '__main__':

    json_folder = Path(r"G:\My Drive\הקוצ'ינים הצעירים\israeli-Indian Hackathon\Info for Participants\VR steps\Data\New data 18.08.25")
    json_name = r"Copy of pedisol_segment_0-603"

    json_path = json_folder.joinpath(json_name + '.json')
    data = load_json(json_path)

    # List of samples where each sample is a dict with the following keys:
    # id (string)
    # Session (string)
    # Expire (seconds number, nanoseconds number)
    # R (list)
    # L (list)
    # T (number)

    t = np.sort([data[i]["T"] for i in range(len(data))])
    R = np.vstack([data[i]["R"] for i in range(len(data))])
    L = np.vstack([data[i]["L"] for i in range(len(data))])
