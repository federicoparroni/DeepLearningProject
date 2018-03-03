import numpy as np
import math
import matplotlib.pyplot as plt


def plot(images, labels, count):
    fig = plt.figure()

    print(len(images))
    print(len(labels))

    for i in range(count):
        a = fig.add_subplot(math.ceil(count ** .5), math.ceil(count ** .5), i+1)
        plt.imshow(np.squeeze(images[i], axis=2), 'gray')
        a.set_title(labels[i])

    plt.show()









