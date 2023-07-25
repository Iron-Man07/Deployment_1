"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
# import datetime

# import torch
import cv2
# import numpy as np
# import tensorflow as tf
# from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
# import subprocess
# from subprocess import Popen
# import re
# import requests
# import shutil
import time
# import glob

from ultralytics import YOLO

app = Flask(__name__)





@app.route("/")
def hello_world():
    return render_template('index.html')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)





    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            global imgpath
            predict_img.imgpath = f.filename
            print("Filename", f.filename)
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower() 
            print("File Extension: ", file_extension)   
            if file_extension == 'jpg' or 'jpeg' or 'JPG':
                img = cv2.imread(filepath)
                
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))
                
                # Perform the Detection
                yolo = YOLO('best.pt') 
                # detections = yolo.predict(image, save= True)
                detections = yolo.predict(img, save= True)
                print("Filepath: ",filepath)   
                return display(f.filename)
                
                    
                       
                # process = Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True)
                # process.wait()
                
                
            elif file_extension == 'mp4':
                video_path = filepath # Replace with your video path
                cap= cv2.VideoCapture(video_path)
                
                # get Video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Define the codec and create VideoWriter Object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
                # Initialize the YOLOv8 model here
                model = YOLO('best.pt')
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Do yolov8 detection on the frame here
                    results = model(frame, save= True)
                    print(results)
                    cv2.waitKey(1)
                    
                    res_plotted = results[0].plot()
                    cv2.imshow("results", res_plotted)
                    
                    # Write the frame to the output video
                    out.write(res_plotted)
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                return video_feed()        
                
                
                
                # process = Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True)
                # process.communicate()
                # process.wait()

            
    # folder_path = 'C:/Kas/temp/runner/w/Subhan_Khan/CV/Deployment/Object-Detection-Web-App-Using-YOLOv8-and-Falsk/runs/detect'
    # folder_path = 'C:/Kas/temp/runner/w/Subhan_Khan/CV/Deployment/runs/detect'
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"
    
    
    
#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    # folder_path = 'C:/Kas/temp/runner/w/Subhan_Khan/CV/Deployment/Object-Detection-Web-App-Using-YOLOv8-and-Falsk/runs/detect'
    # folder_path = 'C:/Kas/temp/runner/w/Subhan_Khan/CV/Deployment/runs/detect'
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory) 
    
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)
    
    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()
     
    # filename = predict_img.imgpath
    # file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        # return send_from_directory(directory,filename,environ)
        return send_from_directory(directory,latest_file,environ)            # Shows the result in seperate tab

    # elif file_extension == 'mp4':
    #     return render_template('index.html')

    else:
        return "Invalid file format"    
    
    
def get_frame():
    # folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    # filename = predict_img.imgpath    
    # image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    # video = cv2.VideoCapture(image_path)  # detected video path
    # #video = cv2.VideoCapture("video.mp4")
    
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)         # Detected video path
    
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds:    
        
        
# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("Function Called")
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')         



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    # model = torch.hub.load('.', 'custom','C:/Kas/temp/runner/w/Subhan_Khan/CV/Deployment/Object-Detection-Web-App-Using-YOLOv8-and-Falsk/best.pt', source='local')
    # model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

