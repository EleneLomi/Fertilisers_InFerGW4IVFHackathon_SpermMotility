import cv2
import numpy as np

# Read the video file
cap = cv2.VideoCapture("tests/data/simple_video/sample1_vid1_sperm14_id17.mp4")
#
# cap = cv2.VideoCapture(
#  "/Users/elenelominadze/Downloads/Hackathon_data/SpermDB/Sample3/sample3_vid10_sperm16_id31.mp4"
# )

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Select the first frame and detect features to track
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(
    old_gray, maxCorners=2, qualityLevel=0.05, minDistance=40, blockSize=7
)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ##  ROI_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[100:200,100:200]

    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if (
        np.any(p1[:, 0, 0] <= 5)
        or np.any(p1[:, 0, 0] >= frame_gray.shape[1] - 5)
        or np.any(p1[:, 0, 1] <= 5)
        or np.any(p1[:, 0, 1] >= frame_gray.shape[0] - 5)
    ):
        # If any points are outside the frame, reinitialize points within the ROI
        frame_exclude_center = np.array(old_gray)[:, 25:]
        p0 = cv2.goodFeaturesToTrack(
            frame_exclude_center,
            maxCorners=1,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )
        p1 = cv2.goodFeaturesToTrack(
            frame_exclude_center,
            maxCorners=1,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )
        good_new = p1
        good_old = p0
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray[100:200,100:200], frame_gray[100:200,100:200],None, None, **lk_params)
        # p1 = cv2.goodFeaturesToTrack(old_gray[100:200,100:200], maxCorners=2, qualityLevel=0.3, minDistance=40, blockSize=17)
    else:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Select good points (those with status = 1)
    #  good_new = p1[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    #    mask = cv2.line(mask, (a, b), (150+c, 150+d), (0, 255, 0), 2)
    # frame = cv2.circle(frame, (150+a, 150+b), 5, (0, 0, 255), -1)
    #    frame = cv2.circle(frame, (150, 150), 5, (0, 0, 255), -1)
    # Overlay the tracks on the frame
    img = cv2.add(frame, mask)

    # Update the previous frame and points
    old_gray = frame_gray.copy()

    # Check if any tracked points are outside the frame boundaries
    p0 = good_new.reshape(-1, 1, 2)

    # Display the result
    cv2.imshow("Object Tracking", img)

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
