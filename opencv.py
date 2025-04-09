# import cv2
# cam = cv2.VideoCapture(0)

# # Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# while True:
#     ret, frame = cam.read()

#     # Write the frame to the output file
#     out.write(frame)

#     # Display the captured frame
#     cv2.imshow('Camera', frame)

#     # Press 'q' to exit the loop
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the capture and writer objects
# cam.release()
# out.release()
# cv2.destroyAllWindows()
# importing the required modules 
import cv2 
import numpy as np 

# capturing from the first camera attached 
cap = cv2.VideoCapture(0) 

# will continue to capture until 'q' key is pressed 
while True: 
	ret, frame = cap.read() 

	# Capturing in grayscale 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	
	cv2.imshow('frame', frame) 
	cv2.imshow('gray', gray) 

	# Program will terminate when 'q' key is pressed 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# Releasing all the resources 
cap.release() 
cv2.destroyAllWindows() 
