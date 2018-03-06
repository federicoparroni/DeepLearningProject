from keras.models import load_model
import pygame.camera
import pygame
from pygame.locals import *
import numpy as np
import FaceExtractionPipeline
import skimage.io
import matplotlib.pyplot as plt
import skimage.transform


def StartDemo(ref, model):
    bp = 'trained_model/'
    model = load_model(bp + model)

    # ref_img = FaceExtractionPipeline.FaceExtractionPipelineImage(skimage.io.imread(ref))

    DEVICE = '/dev/video0'
    SIZE = (480, 480)

    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode(SIZE, 0)
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()
    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    #while capture:

    img_data = pygame.surfarray.array3d(camera.get_image())

    # plt.imshow(img_data)
    # plt.show()

    img_data = skimage.transform.rotate(img_data, 270)

    img_data_pipelined = FaceExtractionPipeline.FaceExtractionPipelineImage(img_data)


        # if img_data_pipelined is not None:
        #
        #     print(img_data_pipelined.shape)
        #
        #     inp = np.concatenate((ref_img, img_data_pipelined))
        #     predicted_label = model.predict(inp)
        #     print(predicted_label)
        #
        # display.blit(screen, (0, 0))
        # pygame.display.flip()
        # for event in pygame.event.get():
        #     print(event.type)
        #     # if event.type == "QUIT":
        #     #     capture = False
        #     # elif event.type == KEYDOWN and event.key == K_s:
        #     #     pygame.image.save(screen, FILENAME)
    camera.stop()
    pygame.quit()

# StartDemo('/home/giovanni/Immagini/Webcam/sample.jpg', 'my_model.h5')
