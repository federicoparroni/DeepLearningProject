from keras.models import load_model
import FaceExtractionPipeline
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import cv2
import tensorflow as tf
import LoadData
import threading


class Demo:

    cap=None
    graph=None
    model=None
    ref_img=None

    def ElaborateImagesAndMakePredition(self, inp_img):
        img_data_pipelined = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(inp_img)

        if img_data_pipelined is not None:
            # plt.imshow(self.ref_img, 'gray')
            # plt.show()
            # plt.imshow(img_data_pipelined, 'gray')
            # plt.show()

            inp = LoadData.MergeImages(self.ref_img, img_data_pipelined)
            inp = np.expand_dims(inp, axis=0)
            # inp = np.expand_dims(inp, axis=3)

            with self.graph.as_default():
                predicted_label = self.model.predict(inp)

            print(('same' if predicted_label[0, 1] > 0.5 else 'wrong') + str(predicted_label))


    def OneFrameComputation(self):
        threading.Timer(0.5, self.OneFrameComputation).start()

        # read frame
        ret, frame = self.cap.read()

        # do the prediction in a different thread
        t = threading.Thread(target=self.ElaborateImagesAndMakePredition, args=(frame,))
        t.setDaemon(True)
        t.start()


    def StartDemo(self, ref, model):
        # initialization of class variables

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640);
        self.cap.set(4, 480);

        self.graph = tf.get_default_graph()

        bp = 'trained_model/'
        self.model = load_model(bp + model)

        self.ref_img = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(skimage.io.imread(ref))

        self.OneFrameComputation()

demo=Demo()
demo.StartDemo('/home/giovanni/Immagini/Webcam/io.jpg', '2018-03-08 01:33:11.h5')