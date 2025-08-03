#!/usr/bin/env python3
import cv2, numpy as np, pathlib, tqdm

CAL_DIR = pathlib.Path("calibration")
CAL_DIR.mkdir(exist_ok=True)
PATTERN = (9, 6)
square_mm = 25.0

objp = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * square_mm
objpoints, imgpoints = [], []

cap = cv2.VideoCapture(0)
print("📸  Press SPACE to capture a good chessboard view, Q to quit.")
count = 0
needed = 20

while count < needed:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    vis = frame.copy()
    if ret:
        cv2.drawChessboardCorners(vis, PATTERN, corners, ret)
    cv2.imshow("calibration", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.imwrite(str(CAL_DIR / f"cal_{count:02d}.jpg"), frame)
        count += 1
        print(f"Captured {count}/{needed}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if count >= 9:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez(CAL_DIR / "calib.npz", mtx=mtx, dist=dist)
    print("✅  Calibration saved.")
else:
    print("❌  Too few images.")
