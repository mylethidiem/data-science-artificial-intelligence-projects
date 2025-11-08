import csv
import os  # interact with os

import cv2  # OpenCV for images and videos processing
import mediapipe as mp  # Google's framework for hand recognition and landmarks
import numpy as np
import yaml


def is_handsign_character(char: str):
    """
    Check if the input character is a lowercase from 'a' to 'z' or a space
    """
    return ord("a") <= ord(char) < ord("q") or char == " "


def label_dict_from_config_file(relative_path: str):
    """
    Read hand_gestures.yaml file and return a dictionary contain gesture labels
    """
    label_tag = []
    with open(relative_path, "r") as file:
        label_tag = yaml.full_load(file)["gestures"]
    return label_tag


class HandDatasetWriter:
    def __init__(self, filepath) -> None:
        self.csv_file = open(filepath, "a")
        self.csv_writer = csv.writer(
            self.csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

    def add_row(self, hand, label):
        self.csv_writer.writerow([label, *np.array(hand).flatten().tolist()])

    def close_file(self):
        self.csv_file.close()


class HandLandmarksDetector:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )

    def detect_hand(self, frame):
        """
        This function is used to detect and extract hand landmarks from a video frame,
        and draw these landmarks onto the image
        """
        hands = []  # empty list to save info about hand

        # flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # make a copy of frame to draw 'annotation'
        annotated_image = frame.copy()

        # convert BGR to RGB and process frame to detect hands
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #
        if results.multi_hand_landmarks is not None:
            hand_conn_style = (
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            hand_landmarks_style = (
                self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                #
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmarks_style,
                    connection_drawing_spec=hand_conn_style,
                )

                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend(x, y, z)
                hands.append(hand)
        return hands, annotated_image


def run(data_path, sign_img_path, split="val", resolution=(1280, 720)):
    """
    Purpose: Colect hand gesture data

    Parameters:
    - data_path: CSV data storage path
    - sign_img_path: Gesture image storage path.
    NOTE: Only the last image of each class will be saved to the path
    for users to check if the image and class are correct
    - split =  Data type(train, val, test)
    - resolution: Resolution of the webcam
    """
    # 0. Prepare
    # Initialize hand detector object, webcam capture
    hand_detector = HandLandmarksDetector()
    cam = cv2.VideoCapture(0)
    # Set webcam resolution width, height
    cam.set(3, resolution[0])
    cam.set(4, resolution[1])

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sign_img_path, exist_ok=True)
    print(f"sign_img_path = {sign_img_path}")

    dataset_path = f"./{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)

    current_letter = None
    status_text = None
    # Flag to prevent switching characters while recording
    cannot_switch_char = False

    # Store frame for saving example images
    saved_frame = None

    # 1. Main loop while camera is open
    while cam.isOpened():
        # 1.1 Read frame from camera
        _, frame = cam.read()

        # 1.2. Detect hand landmarks and get annotated image
        hands, annotated_image = hand_detector.detect_hand(frame)

        # 1.3. Set status text based on recording state
        if current_letter is None:
            status_text = "Press a character to record"
        else:
            # Calculate label index from character
            label = ord(current_letter) - ord("a")
            if label == -65:
                status_text = "Recording unknown, press Spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop!"

        # 1.4. Handle keyboard input
        key = cv2.waitKey(1)
        if key == -1:  # No key press
            if current_letter is None:
                # No current letter recording, just skip it
                pass
            else:
                # If hand detected, save landmark data
                if len(hands) != 0:
                    hand = hands[0]
                    hand_dataset.add_row(hand=hand, label=label)
                    saved_frame = frame
        else:  # Some key is pressed
            # Convert this key to character
            key = chr(key)

            if key == "q":
                break  # Quit program
            if is_handsign_character(key):
                if current_letter is None:
                    current_letter = key
                elif current_letter == key:
                    # pressed again, reset the current state?
                    if saved_frame is not None:
                        if label >= 0:
                            # Save example image for gesture
                            cv2.imwrite(
                                f"./{sign_img_path}/{LABEL_TAG[label]}.jpg", saved_frame
                            )
                else:
                    # Prevent switching whiel recording
                    cannot_switch_char = True

        # 1.5. Display warning if trying to switch character
        if cannot_switch_char:
            cv2.putText(
                img=annotated_image,
                text=f"please press {current_letter} again to unbind",  # Text string to display
                org=(0, 450),  # Bottom-left corner coordinates (x,y) of the text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type/style to use
                fontScale=1,  # Font scale factor (size)
                color=(0, 255, 0),  # Text color in BGR format (green)
                thickness=2,  # Thickness of the text strokes
                lineType=cv2.LINE_AA,
            )  # Line type - anti-aliased line for smoother text

        # 1.6. Display the status text
        cv2.putText(
            img=annotated_image,
            text=status_text,
            org=(5, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # 1.7. Show the annotated image
        cv2.imshow(winname=f"{split}", mat=annotated_image)

    # 2. clean up all thing after finishing task
    cv2.destroyAllWindows()


if __name__ == "__main__":
    LABEL_TAG = label_dict_from_config_file("hand_gesture.yaml")
    data_path = "./data"
    sign_img_path = "./sign_imgs2"
    run(data_path, sign_img_path, "train", (1280, 720))
    run(data_path, sign_img_path, "val", (1280, 720))
    run(data_path, sign_img_path, "test", (1280, 720))
