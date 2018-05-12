from keras.models import load_model
import FaceExtractionPipeline
import skimage.io
import skimage.transform
import cv2
import tensorflow as tf
import LoadData
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import ModelBuilder
from ModelBuilder import read_model


class Demo:

    cap = None
    graph = None
    model = None
    ref_img = None

    def ElaborateImagesAndMakePredition(self, inp_img):
        # crop a good percentage of the image in order to gain performances. found a good tradeoff with those values
        img_data_pipelined = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(inp_img,
                                                                                                    math.ceil(np.shape(inp_img)[0]*15/100),
                                                                                                    math.ceil(np.shape(inp_img)[0]*30/100))

        if img_data_pipelined is not None:
            plt.imshow(img_data_pipelined, 'gray')
            plt.show()

            inp = LoadData.MergeImages(self.ref_img, img_data_pipelined)
            inp = np.expand_dims(inp, axis=0)

            #with self.graph.as_default():
            predicted_label = self.model.predict(inp)

            print(('same' if predicted_label[0, 1] > 0.85 else 'wrong') + str(predicted_label))

        self.OneFrameComputation()


    def OneFrameComputation(self):
        # threading.Timer(0.5, self.OneFrameComputation).start()

        # read frame
        ret, frame = self.cap.read()

        self.ElaborateImagesAndMakePredition(frame)

        # do the prediction in a different thread
        #t = threading.Thread(target=self.ElaborateImagesAndMakePredition, args=(frame,))
        #t.setDaemon(True)
        #t.start()


    def StartDemo(self, ref, model_path):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # self.graph = tf.get_default_graph()

        bp = 'trained_model/'

        a = read_model("models/model1.txt")
        modelObject = ModelBuilder.ModelBuilder(a, (80, 80, 2))
        self.model = modelObject.model
        self.model.load_weights(bp+model_path)

        #self.model = load_model(bp+model_path)
        self.ref_img = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(skimage.io.imread(ref))

        plt.imshow(self.ref_img, 'gray')
        plt.show()

        self.OneFrameComputation()

demo=Demo()
demo.StartDemo('/home/edoardo/Pictures/Webcam/3.jpg', '2018-05-12 15:45:47.h5')
