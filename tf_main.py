import os
import csv
import copy
import itertools
import numpy as np
import tensorflow as tf
import cv2 as cv
import mediapipe as mp
from collections import Counter, deque
from tensorflow.keras import layers, models
from utils import CvFpsCalc
from new_model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from new_model.point_history_classifier.point_history_classifier import PointHistoryClassifier

class HandGestureRecognition:
    def __init__(self, 
                 dataset_dir='data/newDataset', 
                 image_size=(224, 224), 
                 batch_size=32, 
                 num_classes=7):
        # Mediapipe hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Transfer learning model setup
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dataset_dir = dataset_dir
        self.model = self.create_transfer_learning_model()

        # Keypoint and point history tracking
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Load labels
        self.load_labels()

    def load_labels(self):
        # Load labels for keypoint and point history classifiers
        with open('new_model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        
        with open('new_model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    def load_data_from_directory(self):
        """Load dataset from directory for transfer learning."""
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='int',
            shuffle=True
        )
        return dataset

    def create_transfer_learning_model(self):
        """Create transfer learning model using MobileNetV2."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_landmark(self, landmark_list):
        """Preprocess landmarks for keypoint classification."""
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0]
        temp_landmark_list = [[x - base_x, y - base_y] for x, y in temp_landmark_list]

        # Convert to a one-dimensional list and normalize
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, temp_landmark_list))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

        return temp_landmark_list

    def preprocess_point_history(self, point_history):
        """Preprocess point history for gesture classification."""
        # Ensure point_history is a list of numbers
        point_history = list(point_history)
        
        # Pad or truncate point history to exactly 32 points
        if len(point_history) > 32:
            point_history = point_history[-32:]
        else:
            point_history.extend([0] * (32 - len(point_history)))
        
        return point_history

    def recognize_hand_gesture(self, frame):
        """Recognize hand gestures using Mediapipe and transfer learning."""
        # Convert frame to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process frame with Mediapipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to list of points
                landmark_list = [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] 
                                 for lm in hand_landmarks.landmark]
                
                # Preprocess landmarks for keypoint classification
                processed_landmarks = self.preprocess_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(processed_landmarks)
                
                # Preprocess point history for gesture classification
                processed_point_history = self.preprocess_point_history(self.point_history)
                finger_gesture_id = self.point_history_classifier(processed_point_history)

                # Use transfer learning model for prediction
                # Prepare image for transfer learning model
                resized_frame = cv.resize(frame, self.image_size)
                normalized_frame = resized_frame / 255.0
                input_frame = np.expand_dims(normalized_frame, axis=0)
                
                # Predict gesture
                prediction = self.model.predict(input_frame)
                gesture_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                return gesture_id, confidence, landmark_list, hand_sign_id, finger_gesture_id
        
        return None, None, None, None, None

    def run_recognition(self):
        """Real-time hand gesture recognition."""
        cap = cv.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv.flip(frame, 1)
            
            # Recognize gesture
            gesture_id, confidence, landmark_list, hand_sign_id, finger_gesture_id = self.recognize_hand_gesture(frame)
            
            if gesture_id is not None:
                # Draw bounding box and gesture info
                cv.putText(frame, 
                           f"Gesture: {self.keypoint_classifier_labels[hand_sign_id]} ({self.point_history_classifier_labels[finger_gesture_id]}) (Conf: {confidence:.2f})", 
                           (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            
            fps = self.cvFpsCalc.get()
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv.imshow('Hand Gesture Recognition', frame)
            
            # Exit on 'q' key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()

def main():
    recognizer = HandGestureRecognition()
    recognizer.run_recognition()

if __name__ == '__main__':
    main()