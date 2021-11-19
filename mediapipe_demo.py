import cv2
import numpy as np
import mediapipe as mp
import sys
import time

iteration = 1000
file_path = sys.argv[1]
static = True if int(sys.argv[2]) else False
model = int(sys.argv[3])

frame = cv2.imread(file_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# True - Image
# False - Video
holistic = mp_holistic.Holistic(static_image_mode=static, min_detection_confidence=0.5, model_complexity=model)

start_time = time.time()
for i in range(iteration):
	results = holistic.process(frame)
elapsed_time = time.time() - start_time
print(iteration / elapsed_time)

#print("Time taken:", elapsed_time)
#print("FPS:", iteration / elapsed_time)



