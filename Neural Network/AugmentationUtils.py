import Augmentor
import os

# works recursively from the root path
def AugmentDataFromPath(path):
    entries = os.scandir(path)
    p = Augmentor.Pipeline(path,
                           output_directory='.')
    p.flip_left_right(probability=0.5)
    p.skew_left_right(probability=1, magnitude=0.20)

    for entry in entries:
        if entry.is_dir():
            AugmentDataFromPath(entry.path)
        else:
            p.sample(5*(len([name for name in entries])+1))
            break
