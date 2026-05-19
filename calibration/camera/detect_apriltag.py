import argparse

import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

def parse_args():
    parser = argparse.ArgumentParser(description="Detect AprilTags from a webcam or capture device.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=4,
        help="OpenCV camera index passed to cv.VideoCapture (default: 4).",
    )
    return parser.parse_args()


args = parse_args()

cap = cv.VideoCapture(args.camera_index)


cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# --- AprilTag Detector ---
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

def draw_table(frame, tags):
    h, w, _ = frame.shape
    table_width = 400

    # Create a new larger frame with space for the table
    new_frame = np.zeros((h, w + table_width, 3), dtype=np.uint8)

    # Copy the camera feed on the left side
    new_frame[:, :w] = frame

    # Draw the table background on the right side
    x0 = w  # start of table
    cv.rectangle(new_frame, (x0, 0), (x0 + table_width, h), (30, 30, 30), -1)

    # Header
    y = 30
    cv.putText(new_frame, "AprilTag Pose Table", (x0 + 20, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30
    cv.line(new_frame, (x0 + 10, y), (x0 + table_width - 10, y), (255, 255, 255), 1)
    y += 30

    header = "ID | X (m)  Y (m)  Z (m) | R  P  Y (deg)"
    cv.putText(new_frame, header, (x0 + 15, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 20
    cv.line(new_frame, (x0 + 10, y), (x0 + table_width - 10, y), (100, 100, 100), 1)
    y += 20

    # Add each tag’s data
    for tag in tags:
        t = tag.pose_t.flatten()
        R = tag.pose_R
        euler = rotationMatrixToEulerAngles(R)
        text = f"{tag.tag_id:2d} | {t[0]:5.2f} {t[1]:5.2f} {t[2]:5.2f} | {euler[0]:5.1f} {euler[1]:5.1f} {euler[2]:5.1f}"
        cv.putText(new_frame, text, (x0 + 15, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        if y > h - 30:
            break

    return new_frame


# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        tag_size=0.04, # if tag size is different, can change this
        camera_params=([724.693, 591.333, 628.787, 408.007])
    )

    # Draw tag boundaries and centers
    for tag in tags:
        corners = tag.corners
        for i in range(4):
            pt1 = tuple(int(x) for x in corners[i])
            pt2 = tuple(int(x) for x in corners[(i + 1) % 4])
            cv.line(frame, pt1, pt2, (0, 255, 0), 2)
        center = tuple(int(x) for x in tag.center)
        cv.circle(frame, center, 4, (0, 0, 255), -1)
        cv.putText(frame, str(tag.tag_id), (int(center[0] + 10), int(center[1] - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw table on right side
    frame = draw_table(frame, tags)

    cv.imshow('AprilTag Detection with Pose Table', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
