"""
Main file for gesture recognition and light control
"""

# import os  # Import os module for interacting with the operating system
import time  # Import time module for time-related functions

import cv2  # Import OpenCV module for image processing
import mediapipe as mp  # Import MediaPipe module for hand tracking
import numpy as np  # Import NumPy module for numerical processing
import torch
import yaml  # Import PyYAML module for YAML file processing
from controller import ModbusMaster  # Import ModbusMaster class from controller.py
from torch import nn


class HandLandmarksDetector:
    # Class for detecting hand landmarks using MediaPipe
    def __init__(self) -> None:
        self.mp_drawings_utils = (
            mp.solutions.drawing_utils
        )  # MediaPipe drawing utilities
        self.mp_drawings_styles = (
            mp.solutions.drawing_styles
        )  # MediaPipe drawing styles
        self.mp_hands = mp.solutions.hands  # MediaPipe hands module
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )  # Initialize the hands module

    def detect_hand(self, frame):
        # Detect hand landmarks in the frame
        hands = []
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        annotated_image = frame.copy()  # Create a copy of the frame

        results = self.detector.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # Process the frame

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawings_utils.draw_landmarks(
                    annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )  # Draw landmarks and connections
                self.mp_drawings_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawings_styles.get_default_hand_landmarks_style(),
                    self.mp_drawings_styles.get_default_hand_connections_style(),
                )
                for landmark in hand_landmarks.landmark:
                    hand.append(
                        [landmark.x, landmark.y, landmark.z]
                    )  # Append the landmark coordinates to the hand list
                hands.append(hand)
        return hands, annotated_image


class NeuralNetwork(nn.Module):
    # Class for the neural network for gesture recognition
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # Flatten layer to convert input to 1D
        list_label = label_dict_from_config_file(
            "hand_gesture.yaml"
        )  # Load gesture labels from YAML file
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(
                63, 128
            ),  # Linear layer with 63 input features and 128 output features
            nn.ReLU(),  # ReLU activation function
            nn.BatchNorm1d(128),  # Batch normalization layer
            nn.Linear(128, 128),  # Linear layer with 128 input and output features
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.4),  # Dropout layer with 40% dropout rate
            nn.Linear(128, 128),  # Linear layer with 128 input and output features
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.4),  # Dropout layer with 40% dropout rate
            nn.Linear(128, 128),  # Linear layer with 128 input and output features
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.6),  # Dropout layer with 60% dropout rate
            nn.Linear(
                128, len(list_label)
            ),  # Linear layer with 128 input features and output features equal to number of labels
        )

    def forward(self, x):
        # Forward pass through the neural network
        """
        In the line logits = self(x), self(x)
        is calling the forward method of the NeuralNetwork class.
        In PyTorch, when you call an instance of a nn.Module
        (which NeuralNetwork inherits from) with some input x,
        it internally calls the forward method of that instance.
        In this class, the forward method defines
        how the input x is processed through the network layers.
        When you call self(x), it is equivalent to calling self.forward(x),
        which processes the input x through the network and returns the output logits.
        """
        x = self.flatten(x)  # Flatten the input
        logits = self.linear_relu_stack(x)  # Pass the input through the neural network
        return logits

    def predict(self, x, threshold=0.5):
        # Predict the output of the neural network
        logits = self(x)
        softmax = nn.Softmax(dim=1)(logits)  # Apply softmax function to the output
        chosen_index = torch.argmax(
            softmax, dim=1
        )  # Get the index of the maximum value
        return torch.where(
            softmax > threshold, chosen_index, -1
        )  # Return the index if the value is greater than the threshold, otherwise return -1

    def predict_with_known_class(self, x):
        # Predict the output of the neural network with known classes
        logits = self(x)
        softmax = nn.Softmax(dim=1)(logits)  # Apply softmax function to the output
        # or using Pytorch's directly in sensor
        # softmax = torch.softmax(logits, dim=1)

        # torch.argmax(dim=1) Returns the index of the largest value in the row (class) direction
        predicted_classes = torch.argmax(softmax, dim=1)

        return softmax, predicted_classes

    def score(self, logits):
        # Calculate the score of the output
        return -torch.amax(logits, dim=1)  # Return the maximum value of the logits


def label_dict_from_config_file(relative_path):
    # Load gesture labels from a YAML file
    with open(relative_path, "r") as file:
        label_dict = yaml.full_load(file)["gestures"]  # Load the YAML file
    return label_dict


class LightGesture:
    # Class for controlling lights using hand gestures
    def __init__(self, model_path, device=False) -> None:
        self.device = device  # Flag to indicate if a device is connected
        self.height = 720  # Height of the video frame
        self.width = 1280  # Width of the video frame

        self.detector = HandLandmarksDetector()  # Initialize hand landmarks detector
        self.status_text = None  # Variable to store the status text
        self.signs = label_dict_from_config_file(
            "hand_gesture.yaml"
        )  # Load gesture labels from YAML file
        self.classifier = NeuralNetwork()  # Initialize the neural network classifier
        self.classifier.load_state_dict(
            torch.load(model_path)
        )  # Load the trained model
        self.classifier.eval()  # Set the model to evaluation mode

        if self.device:
            self.controller = (
                ModbusMaster()
            )  # Initialize the Modbus controller if a device is connected
        self.light1 = False  # Status of light 1
        self.light2 = False  # Status of light 2
        self.light3 = False  # Status of light 3

    def light_device(self, img, lights):
        # Append a white rectangle at the bottom of the image to indicate light status
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255

        # Draw a red border around the rectangle
        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)

        # Calculate circle positions
        circle_radius = int(0.45 * rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))

        # Draw the circles to indicate light status
        on_color = (0, 255, 255)
        off_color = (0, 0, 0)
        colors = [off_color, on_color]
        circle_centers = [circle1_center, circle2_center, circle3_center]
        for cc, light in zip(circle_centers, lights):
            color = colors[int(light)]
            cv2.circle(white_rect, cc, circle_radius, color, -1)

        # Append the white rectangle to the bottom of the image
        img = np.vstack((img, white_rect))
        return img

    def run(self):
        # Main function to capture video and detect gestures
        cam = cv2.VideoCapture(0)  # Open the webcam
        cam.set(3, 1280)  # Set the width of the video frame
        cam.set(4, 720)  # Set the height of the video frame
        while cam.isOpened():
            _, frame = cam.read()  # Read a frame from the webcam

            hand, img = self.detector.detectHand(
                frame
            )  # Detect hand landmarks in the frame
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(
                        np.array(hand[0], dtype=np.float32).flatten()
                    ).unsqueeze(
                        0
                    )  # Convert the hand landmarks to a tensor
                    class_number = self.classifier.predict(
                        hand_landmark
                    ).item()  # Predict the gesture class
                    if class_number != -1:
                        self.status_text = self.signs[class_number]

                        if self.status_text == "light1":
                            if self.light1 is False:
                                print("lights on")
                                self.light1 = True
                                if self.device:
                                    self.controller.switch_actuator_1(True)
                        elif self.status_text == "light2":
                            if self.light2 is False:
                                self.light2 = True
                                if self.device:
                                    self.controller.switch_actuator_2(True)
                        elif self.status_text == "light3":
                            if self.light3 is False:
                                self.light3 = True
                                if self.device:
                                    self.controller.switch_actuator_3(True)
                        elif self.status_text == "turn_on":
                            # self.light1 = self.light2 = self.light3 = True
                            if self.light1 and self.light2 and self.light3:
                                pass
                            else:
                                self.light1 = self.light2 = self.light3 = True
                                if self.device:
                                    self.controller.switch_actuator_1(self.light1)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_2(self.light2)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_3(self.light3)
                        elif self.status_text == "turn_off":
                            if not self.light1 and not self.light2 and not self.light3:
                                pass
                            else:
                                self.light1 = self.light2 = self.light3 = False
                                if self.device:
                                    self.controller.switch_actuator_1(self.light1)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_2(self.light2)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_3(self.light3)
                    else:
                        self.status_text = "undefined command"
            else:
                self.status_text = None

            img = self.light_device(
                img, [self.light1, self.light2, self.light3]
            )  # Update the image with light status

            cv2.putText(
                img,
                self.status_text,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.namedWindow("window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("window", 1920, 1080)
            cv2.imshow("window", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "./models/model_02-11 15_19_NeuralNetwork_best"
    light = LightGesture(model_path, device=False)
    light.run()
