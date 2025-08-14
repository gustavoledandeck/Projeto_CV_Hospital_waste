#!/usr/bin/env python3
import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torchvision import models
import pickle
from collections import deque



# Func para ajudar a achar a imagem p/ deteccao
def get_iou(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Loading ---
num_classes = 4
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model.load_state_dict(torch.load("models/best_resnet18.pth", map_location=device))
model = model.to(device).eval()

# --- Class names ---
try:
    with open('models/class_names.pkl', 'rb') as f:
        classes = pickle.load(f)
    print(f"Loaded class names: {classes}")
except FileNotFoundError:
    print("Warning: 'models/class_names.pkl' not found. Using default class names.")
    classes = ['high', 'low', 'medium', 'normal']

# --- Image Transformations for the Classifier ---
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_classify = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_left = cv2.VideoWriter('Inferencia_Rangers.mp4', fourcc, fps, (frame_width, frame_height))
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


tracked_objects = []

next_object_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Deteccao usa parte do lab, com blur e canny, acrescentando a findContours que é nova
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    if contours:
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 50 and h > 50:
                    current_detections.append([x, y, x + w, y + h])

    # --- Match current detections with tracked objects ---
    unmatched_detections = list(range(len(current_detections)))
    
    for obj in tracked_objects:
        obj['matched'] = False
        best_iou = 0
        best_match_idx = -1

        for i in unmatched_detections:
            iou = get_iou(obj['box'], current_detections[i])
            if iou > best_iou:
                best_iou = iou
                best_match_idx = i
        
        
        if best_iou > 0.4: 
            matched_box = current_detections[best_match_idx]
            
            obj['box'] = (np.array(obj['box']) * 0.8 + np.array(matched_box) * 0.2).astype(int)
            obj['unseen_frames'] = 0
            obj['matched'] = True
            unmatched_detections.remove(best_match_idx)

    tracked_objects = [obj for obj in tracked_objects if obj['unseen_frames'] < 10]  ###-->> pra tirar obj/diminuir spam
    for obj in tracked_objects:
        if not obj['matched']:
            obj['unseen_frames'] += 1

    for i in unmatched_detections:
        box = current_detections[i]
        x1, y1, x2, y2 = box
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: continue
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = tf_classify(image=roi_rgb)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor)
            probabilities = F.softmax(out, dim=1)
            conf, idx = torch.max(probabilities, 1)
            
            #valor para manter o objeto detecto, só muda caso tenha detectado outro no cena > 0.7
            if conf.item() > 0.7:
                label = classes[idx.item()]
                tracked_objects.append({
                    'id': next_object_id,
                    'box': box,
                    'label': label,
                    'confidence': conf.item(),
                    'unseen_frames': 0,
                    'matched': True
                })
                next_object_id += 1

    #Display da box
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['box']
        label = obj['label']
        conf = obj['confidence']
        
        color = (0, 255, 0) # Green
        if conf < 0.85: color = (0, 255, 255) # Yellow
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label.capitalize()} ({conf:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
	
    #cv2.imshow("Hospital Waste Detection", frame)
    cv2.imshow("Hospital Waste Detection - Rangers", frame)
    out_left.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_left.release()
cv2.destroyAllWindows()

