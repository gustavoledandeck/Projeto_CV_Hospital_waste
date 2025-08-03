#!/usr/bin/env python3
import cv2, numpy as np, pathlib

CAL_DIR = pathlib.Path("calibration")
CAL_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)
frames = []
print("🔧 Move the camera around a textured surface. Press SPACE 30×, Q to finish.")
count = 0
needed = 30
while count < needed:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("calibration", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        frames.append(frame)
        count += 1
        print(f"Frame {count}/{needed}")
    elif k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

if count < 9:
    print("❌  Not enough frames — using identity calibration.")
    np.savez(CAL_DIR / "calib.npz", mtx=np.eye(3), dist=np.zeros(5))
else:
    # use regular camera calibration on random images (works for small distortion)
    gray_imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = gray_imgs[0].shape
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    objp = np.zeros((1, 7*10, 3), np.float32)  # 7×10 dummy inner corners
    objp[0, :, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)

    obj_points = [objp] * len(gray_imgs)
    img_points = []

    for g in gray_imgs:
        ret, corners = cv2.findChessboardCorners(g, (7, 10), None)
        if ret:
            corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)

    if len(img_points) < 9:
        print("❌  Couldn’t find enough virtual corners — using identity.")
        np.savez(CAL_DIR / "calib.npz", mtx=np.eye(3), dist=np.zeros(5))
    else:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), K, D)
        np.savez(CAL_DIR / "calib.npz", mtx=K, dist=D)
        print("✅  Auto-calibration complete.")
