import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize mediapipe hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)


def detect_gesture(hand_landmarks):
    # Extract the coordinates of relevant landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate distances to determine gestures
    thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    thumb_middle_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    index_middle_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    palm_width = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([pinky_tip.x, pinky_tip.y]))

    # Detect gestures based on distances
    if thumb_tip.y < index_tip.y and thumb_middle_dist > thumb_index_dist:
        return "thumbs_up"
    elif thumb_tip.y > index_tip.y and thumb_middle_dist > thumb_index_dist:
        return "thumbs_down"
    elif index_tip.y < middle_tip.y and index_middle_dist > palm_width * 0.5:
        return "forefinger_up"
    elif index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y:
        return "two_fingers_up"
    elif index_tip.y > middle_tip.y:
        return "finger_down"
    elif all(tip.y < wrist.y for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
        return "full_palm"
    else:
        return None


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            if gesture == "thumbs_up":
                pyautogui.press('playpause')  # Simulate play/pause key press
                print("Gesture: Play")
            elif gesture == "thumbs_down":
                pyautogui.press('stop')  # Simulate stop key press
                print("Gesture: Stop")
            elif gesture == "forefinger_up":
                pyautogui.press('volumeup')  # Simulate volume up key press
                print("Gesture: Volume Up")
            elif gesture == "finger_down":
                pyautogui.press('volumedown')  # Simulate volume down key press
                print("Gesture: Volume Down")
            elif gesture == "full_palm":
                pyautogui.press(' ')  # Simulate play/pause key press
                print("Gesture: Play/Pause")

    # Display the image
    cv2.imshow('Media Gesture Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()