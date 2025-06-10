import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

def is_hand_closed(landmarks):
    """
    Simple logic: If the tip of fingers is below their base joint (folded), consider it closed.
    Indexes of finger tips: [8, 12, 16, 20]
    Indexes of finger base joints: [6, 10, 14, 18]
    """
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]
    
    closed_fingers = 0
    for tip, dip in zip(finger_tips, finger_dips):
        if landmarks[tip].y > landmarks[dip].y:  # tip below base
            closed_fingers += 1

    return closed_fingers >= 3  # majority of fingers folded

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine open or closed hand
            if is_hand_closed(hand_landmarks.landmark):
                cv2.putText(frame, "✊ Hand Closed", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "🖐️ Hand Open", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Open/Closed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
