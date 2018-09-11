from keras.models import load_model
import FaceExtractionPipeline
import skimage.io
import skimage.transform
import cv2
import LoadData
import threading
import numpy as np
import math
import matplotlib.pyplot as plt

import ModelBuilder
from ModelBuilder import read_model

import time

import tkinter as tk
import PIL.Image, PIL.ImageTk


class Demo:

    cap = None
    graph = None
    model = None
    ref_img = None

    def ElaborateImagesAndMakePredition(self, inp_img):
        # crop a good percentage of the image in order to gain performances. found a good tradeoff with those values

        start_time = time.time()

        img_data_pipelined = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(inp_img,
                                                                                                    math.ceil(np.shape(inp_img)[0]*15/100),
                                                                                                    math.ceil(np.shape(inp_img)[0]*30/100))

        if img_data_pipelined is not None:
            # plt.imshow(img_data_pipelined, 'gray')
            # plt.show()

            inp = LoadData.MergeImages(self.ref_img, img_data_pipelined)
            inp = np.expand_dims(inp, axis=0)

            #with self.graph.as_default():
            predicted_label = self.model.predict(inp)

            print(('same' if predicted_label[0, 1] > 0.975 else 'wrong') + str(predicted_label))

        print("--- %s seconds for a frame---" % (time.time() - start_time))

        # self.OneFrameComputation()


    def OneFrameComputation(self):
        # threading.Timer(0.5, self.OneFrameComputation).start()

        # read frame

        # cv2.imshow('my webcam', frame)

        while True:
            ret, frame = self.cap.read()
            cv2.imshow('my webcam', frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            self.ElaborateImagesAndMakePredition(frame)
        cv2.destroyAllWindows()

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

        a = read_model("models/model99.txt")
        modelObject = ModelBuilder.ModelBuilder(a, (80, 80, 2))
        self.model = modelObject.model
        self.model.load_weights(bp+model_path)

        #self.model = load_model(bp+model_path)
        self.ref_img = FaceExtractionPipeline.SingletonPipeline().FaceExtractionPipelineImage(skimage.io.imread(ref))

        # plt.imshow(self.ref_img, 'gray')
        # plt.show()

        self.OneFrameComputation()

    def Window(self):
        # Set up GUI
        window = tk.Tk()  # Makes main window
        window.wm_title("Face2Face")
        window.config(background="#FFFFFF")

        # Graphics window
        imageFrame = tk.Frame(window, width=300, height=500)
        imageFrame.grid(row=0, column=0, padx=10, pady=2)

        # Capture video frames
        lmain = tk.Label(imageFrame)
        lmain.grid(row=0, column=0)
        cap = cv2.VideoCapture(0)

        #l2 = tk.Label(imageFrame)
        #l2.grid(row=0, column=0)

        def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = PIL.Image.fromarray(cv2image)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)

            # Slider window (slider controls stage position)

        sliderFrame = tk.Frame(window, width=600, height=100)
        sliderFrame.grid(row=600, column=0, padx=10, pady=2)

        show_frame()  # Display 2
        window.mainloop()  # Starts GUI


demo=Demo()
#demo.StartDemo('/home/giovanni/Immagini/Webcam/io.jpg', '2018-07-10 11:27:21/model99.txt_2018-07-10 17:41:54.h5')

#demo.StartDemo('/Users/federico/Desktop/cristiano.jpg', '2018-07-10 11:27:21/model99.txt_2018-07-10 17:41:54.h5')

demo.Window()
