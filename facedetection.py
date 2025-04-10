import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

# def function1():
#     experience = np.array([1,2,3,4,5,6,7,8,9,10])
#     salaries = np.array([15,20,23,29,30,28,35,45,48,50])
#
#     plt.scatter(experience,salaries,color="magenta",label="Salary")
#     plt.plot(experience, salaries, color="magenta")
#     plt.xlabel("Experience (in years)")
#     plt.ylabel("Salary in Thousands ")
#     plt.legend()
#     plt.title("Experience vs Salary")
#     plt.savefig("Experience vs Salary.jpg")
#     plt.show()
#
# function1()

# import pandas as pd
# df = pd.read_csv('Salary_Data.csv')
# print(df)
# print(df.describe())
# print(df.cov())
# print(df.corr())
# print(df.kurtosis())
# word= input("Enter a word")
# nrm=dict()
# for c in word:
#     if c not in nrm:
#         nrm[c]=1
#     else:
#         nrm[c]+=1
# print(nrm)
# import cv2
#
# # read video from a source
# # capture = cv2.VideoCapture('./test.mp4')
# capture = cv2.VideoCapture(0)
#
# # read frames from the video
# index = 0
# while True:
#
#     # read the frame from video
#     ret, frame = capture.read()
#
#     # convert every frame to grayscale
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect edge
#     frame_edge = cv2.Canny(frame, 255, 255)
#
#     # show the frame
#     cv2.imshow("camera-color", frame)
#     cv2.imshow("camera-gray", frame_gray)
#     cv2.imshow("camera-edge", frame_edge)
#
#     # save every frame
#     # cv2.imwrite(f"frames/frame_{index}.jpg", frame)
#     # cv2.imwrite(f"frames_gray/frame_gray_{index}.jpg", frame_gray)
#     # index += 1
#
#     # break the loop or stop the camera when user presses 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
#

# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:

	# reads frames from a camera
	ret, img = cap.read()

	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Detects eyes of different sizes in the input image
		eyes = eye_cascade.detectMultiScale(roi_gray)

		#To draw a rectangle in eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

	# Display an image in a window
	cv2.imshow('img',img)
	

	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
