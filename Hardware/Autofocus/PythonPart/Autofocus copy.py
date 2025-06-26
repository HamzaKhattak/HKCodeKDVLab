import cv2
import time
import numpy as np
import serial
# Open the default camera
ser = serial.Serial('COM4')
time.sleep(1)
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Cannot open camera")
    exit()
nframes = 0
ts = 0
i=0


focusstack = []
ser.write('m3000\n'.encode(encoding="utf-8") )
while True:
    t1=time.time()
    ret, frame = cam.read()
    

    # Display the captured frame
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == ord('j'):
        ser.write('m3000\n'.encode(encoding="utf-8") )
    if cv2.waitKey(1) == ord('k'):
        ser.write('m0\n'.encode(encoding="utf-8") )
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
ser.close()
cam.release()
cv2.destroyAllWindows()