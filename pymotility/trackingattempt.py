import numpy as np
import cv2

# STEP 1: We start by loading the first frame of the video and converting it to grayscale

# Load the first frame of the video
cap = cv2.VideoCapture(
    "/Users/elenelominadze/Downloads/Hackathon_data/SpermDB/Sample3/sample3_vid10_sperm16_id31.mp4"
)
ret, old_frame = cap.read()

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Convert frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# STEP 2: We set the parameters for Lucas-Kanade optical flow (lk_params).

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# STEP 3: We select points to track using the cv2.goodFeaturesToTrack function.

# Select points to track (actually, do this later in the while loop)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)


# STEP 4: Then, we enter a loop where we read each subsequent frame of the video.

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Exclude centerROI

    # STEP 5: For each frame, we calculate the optical flow using cv2.calcOpticalFlowPyrLK.
    p0 = cv2.goodFeaturesToTrack(
        old_gray, mask=None, maxCorners=5, qualityLevel=0.1, minDistance=70
    )

    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )

    # STEP 6: We select the good points from the optical flow calculation.

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # STEP 7: We draw tracks between the old and new points.

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # Calculate displacement
        displacement = np.sqrt((c - a) ** 2 + (d - b) ** 2)

        # Calculate velocity
        velocity = (
            displacement * fps
        )  # Multiply by frame rate to get velocity in pixels per second
    #    print(f"disp of point {i}: {displacement} pixels")
    #    print(f"Velocity of point {i}: {velocity} pixels per second")

    img = cv2.add(frame, mask)

    cv2.imshow("frame", img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
