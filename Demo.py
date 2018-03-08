from keras.models import load_model
import FaceExtractionPipeline
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import cv2
import LoadData


def StartDemo(ref, model):
    # load the pretrained model
    bp = 'trained_model/'
    model = load_model(bp + model)

    # pass the reference image through the pipeline
    ref_img = FaceExtractionPipeline.FaceExtractionPipelineImage(skimage.io.imread(ref))
    ref_img = skimage.color.rgb2gray(ref_img)

    # acquire the video stream
    cap = cv2.VideoCapture(0)

    # set resolution of the acquired image
    cap.set(3, 640);
    cap.set(4, 480);

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # TO Move in the pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_data_pipelined = FaceExtractionPipeline.FaceExtractionPipelineImage(gray)



        if img_data_pipelined is not None:

            # plt.imshow(ref_img, 'gray')
            # plt.show()
            # plt.imshow(img_data_pipelined, 'gray')
            # plt.show()

            inp = LoadData.MergeImages(ref_img, img_data_pipelined)
            inp = np.expand_dims(inp, axis=0)
            #inp = np.expand_dims(inp, axis=3)

            predicted_label = model.predict(inp)
            #print(predicted_label)
            print('same' if predicted_label[0, 1] > 0.5 else 'wrong')


        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


StartDemo('/home/giovanni/Immagini/Webcam/parro.jpg', '2018-03-06 02:18:46.h5')