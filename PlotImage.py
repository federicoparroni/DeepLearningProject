import math
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


def plot(images, labels):
    fig = plt.figure()
    count = len(images)

    for i in range(count):
        a = fig.add_subplot(math.ceil(count ** .5), math.ceil(count ** .5), i+1)
        imgplot = plt.imshow(images[i])
        if len(images) == len(labels):
            a.set_title(labels[i])
        else:
            a.set_title("Image " + str(i))

    plt.show()

