# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import numpy as np
import os
import tensorflow as tf
import cv2
from layers import L1Dist


# Build Layout
class CamApp(App):

    def build(self):
        # Setting Up path
        self.VER_PATH = os.path.join("application_data", "verification_image")
        self.INP_PATH = os.path.join("application_data", "input_image")

        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification = Label(text="Verification Uninitiated", size_hint=(1, 0.1))

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, (1.0 / 33.0))

        # Setting up Siamese_Model
        self.model = tf.keras.models.load_model(
            "siamese_model.keras",
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossEntropy": tf.losses.BinaryCrossentropy,
            },
        )

        return layout

    def update(self, *args):
        # Read frame from OpenCV
        ret, frame = self.capture.read()
        frame = frame[120 : 250 + 120, 200 : 200 + 250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture

    # Load image from the file and Convert it into 100*100 Px
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    # Verification Function
    def verify(self, *args):
        # Specifying detection and verification Threshold
        detection_threshold = 0.9
        verification_threshold = 0.8

        # Capture Image from the Web Cam
        SAVE_PATH = os.path.join(self.INP_PATH, "input_image.jpg")
        ret, frame = self.capture.read()
        frame = frame[120 : 250 + 120, 200 : 200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(self.VER_PATH):
            input_img = self.preprocess(os.path.join(self.INP_PATH, "input_image.jpg"))
            validation_img = self.preprocess(os.path.join(self.VER_PATH, image))

            # Make predictions
            result = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1))
            )
            results.append(result)

        # Detection Threshold : Metrics above which a prediction is considered Positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold : Proportion of Positive Predictions / total Postive Samples
        verification = detection / len(os.listdir(self.VER_PATH))
        verified = verification > verification_threshold

        # Set Verification Content
        self.verification.text = "Verified" if verified == True else "Unverified"

        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.5))

        return results, verified


if __name__ == "__main__":
    CamApp().run()
