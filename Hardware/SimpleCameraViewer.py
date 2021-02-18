import cv2
import PySimpleGUI as sg
window = sg.Window('Demo Application - OpenCV Integration', [[sg.Image(filename='', key='image')],[sg.Image(filename='', key='image2')]], location=(800,400))
cap = cv2.VideoCapture(0)     # Setup the camera as a capture device
cap.set(3,1920)
cap.set(4,1080)
while True:                     # The PSG "Event Loop"
    event, values = window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait
    if event is None:  break                                            # if user closed window, quit
    window.FindElement('image').Update(data=cv2.imencode('.png', cap.read()[1])[1].tobytes()) # Update image in window

