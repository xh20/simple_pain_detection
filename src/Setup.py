import cv2
import sys
import numpy as np
import pyrealsense2 as rs

from src.Trainer import SVM, preprocess_single_hog, KNearest
import dlib


class Setup:
    def __init__(self):

        # using SVM model
        self.svm = SVM()
        self.model_svm = self.svm.load('model_svm.dat')

        # using KNN model
        # self.knn = KNearest()
        # self.model_knn = self.knn.load('model_knn.dat')

        # running
        self.running = True

        # use dlib face landmark
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # number of your own images
        self.num_ur_images = 1000

    def setup_camera(self):
        ## Set up for realsense
        self.pipe = rs.pipeline()
        cfg = rs.config()

        # for D435
        # self.CAM_WIDTH, self.CAM_HEIGHT, CAM_FPS = 848, 480, 30
        # for L515
        self.CAM_WIDTH, self.CAM_HEIGHT, CAM_FPS = 640, 480, 30
        cfg.enable_stream(rs.stream.color, self.CAM_WIDTH, self.CAM_HEIGHT, rs.format.rgb8, 30)  # bgr8 rgb8

        self.pipe.start(cfg)

        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for x in range(5):
            self.pipe.wait_for_frames(3000)

    def crop_face(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        ### using Dlib
        faces = self.detector(gray)

        # only select the first person
        face = faces[0]
        landmarks = self.predictor(image=gray, box=face)
        x1 = landmarks.part(17).x
        y1 = landmarks.part(19).y
        x2 = landmarks.part(26).x
        y2 = landmarks.part(6).y

        self.start_point = (x1, y1)
        self.end_point = (x2, y2)

        return gray[y1:y2, x1:x2]

    def is_pain(self, face):
        pain = False
        img_hog = preprocess_single_hog(face)
        sample = []
        sample.append(img_hog)
        sample_array = np.array(sample)
        prediction = self.svm.predict(sample_array)[0]

        text = "Neutral"
        color = (1, 255, 1)

        # 0: Pain, 1: Neutral
        if prediction == 0:
            color = (1, 1, 255)
            text = "Pain"
            pain = True
        self.image = cv2.rectangle(self.image, self.start_point, self.end_point, color, 2)
        cv2.putText(self.image, "%s" % text, (self.start_point[0], self.start_point[1] - 10), 0, 5e-3 * 100, color, 2)
        return pain

    def run(self, creat_dataset=False, dataset_dir=None):
        # initialization
        self.setup_camera()
        counter = 0

        while self.running:
            cropped_face = None
            # Realsense read frame
            frameset = self.pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            # Validate that frames are valid
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            self.image = color_image

            try:
                cropped_face = self.crop_face()
            except cropped_face is None:
                continue

            resized_face = cv2.resize(cropped_face.copy(), (48, 48), cv2.INTER_AREA)
            if creat_dataset & counter < self.num_ur_images:
                data_dir = dataset_dir + "/own_data/" + str(counter) + ".png"
                print("creating your own data ... ")
                cv2.imwrite(data_dir, resized_face)
                counter += 1
            elif creat_dataset & counter >= self.num_ur_images:
                print("recording is finished")
                print("please move your own data to the correct train and test path")
                exit(1)
            else:
                self.is_pain(resized_face)
                cv2.imshow("is pain?", self.image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
