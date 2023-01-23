# To Capture Frame
import cv2

# To process image array
import numpy as np

# To Load the Pre-trained Model
import tensorflow as tf

camera = cv2.VideoCapture(0)

mymodel = tf.keras.models.load_model('keras_model.h5')

# Infinite loop
while True:

	status , frame = camera.read()

	if status:

		frame = cv2.flip(frame , 1)

		# Resize the frame
		resized_frame = cv2.resize(frame , (224,224))

		# Expanding the dimension of the array along axis 0
		resized_frame = np.expand_dims(resized_frame , axis = 0)

		# Normalizing for easy processing
		resized_frame = resized_frame / 255

		# Getting predictions from the model
		predictions = mymodel.predict(resized_frame)

		# Converting the data in the array to percentage confidence 
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

camera.release()

cv2.destroyAllWindows()