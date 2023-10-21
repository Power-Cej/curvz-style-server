import cv2
import mediapipe as mp
import time


def capture_motion(result):
    # import human body segmentation library
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # calibration factor (pixels per centimeter)
    pixels_per_cm = 2

    # Load an image
    frame = cv2.imread(result)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        # Disable write access to the frame, convert to RGB
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe holistic model
        results = holistic.process(frame)

        # Enable write access to the frame, convert back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            # get the frame height and width
            image_height, image_width, _ = frame.shape

            # identify the shoulder landmarks
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

            # identify hips landmarks
            left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]

            # identify right ankle ladmarks
            right_ankle = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]

            # calculate the shoulder width from left to right and convert it to cm
            shoulder_width_pixels = (
                right_shoulder.x * image_width - left_shoulder.x * image_width)
            shoulder_width_cm = abs(shoulder_width_pixels) / pixels_per_cm

            # calculate the shirt length
            top_length_pixels = (right_shoulder.y *
                                 image_height - right_hip.y * image_height)
            top_length_cm = abs(top_length_pixels) / pixels_per_cm

            # calculate the outside leg
            leg_length_pixels = (right_hip.y * image_height -
                                 right_ankle.y * image_height)
            leg_length_cm = abs(leg_length_pixels) / pixels_per_cm

            shoulder_result = int(shoulder_width_cm)
            shirt_result = int(top_length_cm)
            leg_result = int(leg_length_cm)

            # # display measurements
            print(f'Shoulder Width (cm):---------- {shoulder_width_cm:.2f}')
            print(f'Top Length (cm): {top_length_cm:.2f}')
            print(f'Leg Length (cm): {leg_length_cm:.2f}')
            return {
                'shoulder_width_cm': shoulder_result,
                'top_length_cm': shirt_result,
                'leg_length_cm': leg_result
            }

        # Display the frame with annotations
        cv2.imshow('MediaPipe Holistic', frame)
        cv2.waitKey(0)

    # Close the window after key press
    cv2.destroyAllWindows()
