import cv2


# Function to extract frames
def FrameCapture(path):
	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:
		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		# Saves the frames with frame-count
		cv2.imwrite("./AACM71csS-Q/frame%d.jpg" % count, image)

		count += 1

# FrameCapture('./AACM71csS-Q.mp4')
img = cv2.imread('./AACM71csS-Q/frame1290.jpg')
print(img.shape)
h = img.shape[0]
w = img.shape[1]
start_point = (0.559*w, 0.18666667*h)
end_point = (0.771*w, 1*h)
print(start_point,end_point)

start_point = tuple(map(int,start_point))
end_point = tuple(map(int,end_point))

# Black color in BGR
color = (0, 255, 0)

# Line thickness of -1 px
# Thickness of -1 will fill the entire shape
thickness = 5

image = cv2.rectangle(img, start_point, end_point, color, thickness)

cv2.imshow('Image',img)
cv2.waitKey()
