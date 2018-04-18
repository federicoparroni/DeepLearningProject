import Augmentor
import os
from PIL import Image, ExifTags

# works recursively from the root path
def AugmentDataFromPath(path):
    entries = os.scandir(path)
    p = Augmentor.Pipeline(path, output_directory='.')
    p.flip_left_right(probability=0.5)
    p.skew_left_right(probability=1, magnitude=0.20)

    for entry in entries:
        if entry.is_dir():
            AugmentDataFromPath(entry.path)
        else:
            p.sample(5*(len([name for name in entries])+1))
            break



def FlipImages(path):
    entries = os.scandir(path)
    for entry in entries:
        if entry.is_dir():
            FlipImages(entry.path)
        else:
            try:
                image=Image.open(entry.path)
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                exif=dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                    print(entry.path)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                    print(entry.path)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    print(entry.path)
                image.save(entry.path)
                image.close()

            except (AttributeError, KeyError, IndexError):
                # cases: image don't have getexif
                pass


#FlipImages('/home/edoardo/Git_Projects/DeepLearningProject/Neural Network/1_dataset train')
#FlipImages('/home/edoardo/Git_Projects/DeepLearningProject/Neural Network/2_dataset test')
#AugmentDataFromPath('/home/edoardo/Git_Projects/DeepLearningProject/Neural Network/1_dataset train')
AugmentDataFromPath('/home/edoardo/Desktop/ToAugment')
