# Python 3.6
# OpenCV version 3.3.0

#library==================================================================================================================
import numpy as np
import cv2

#functions================================================================================================================
def centerX(x,w):
	center_x = (x+x+w)/2
	return center_x

def centerY(y,h):
	center_y = (y+y+h)/2
	return center_y

#video capture and background subtraction=================================================================================
cap = cv2.VideoCapture('rsc/carClassifyGood.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

#counters=================================================================================================================
#lanes
lastframeleftleft = 0
lastframeleftright = 0
lastframerightleft = 0
lastframerightright = 0

#text counts
framecount = 0
truck = 0
minibus = 0
carcount = 0
up = 0
down = 0

#font=====================================================================================================================
font = cv2.FONT_HERSHEY_SIMPLEX
#nrow = 640
#ncol = 360

#video rendering==========================================================================================================
while True:
	#frame counter
	framecount = framecount + 1

	ret, frame = cap.read()
	frameCopy = frame
	gaussian = cv2.GaussianBlur(frameCopy, (15,15), 0)
	fgmask = fgbg.apply(gaussian)

	median = cv2.medianBlur(fgmask, 15)

	ret, threshold = cv2.threshold(median, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	kernel = np.ones((5,5), np.uint8)
	erosion = cv2.erode(threshold, kernel, iterations = 5)

	_,contours,_ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	#lines=================================================================================================================
	#left lane, right
	cv2.line(frame, (230,265), (295,265), (0,0,0), 5)
	#left lane, left
	cv2.line(frame, (135,265), (230,265), (0,0,0), 5)

	#right lane, right
	cv2.line(frame, (450,265), (545,265), (0,0,0), 5)
	#left lane, left
	cv2.line(frame, (385,265), (450,265), (0,0,0), 5)


	#draw rectangles for each contour=======================================================================================
	#left lane, right
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)

		#sort the cars based on height size=================================================================================
		if(h>15):
			#left lane left counter
			if ((centerY(y,h) <= 285) and ((centerY(y,h) >= 270))) and ((centerX(x,w) > 105) and ((centerY(x,w) < 230))):
				if((framecount-lastframeleftleft)>=15):
					if (h>70):
						truck+=1
						lastframeleftleft = framecount
						down += 1
					elif (h>45):
						minibus+=1
						lastframeleftleft = framecount
						down += 1
					else:
						carcount += 1
						lastframeleftleft = framecount
						down+=1

			#left lane right counter
			if ((centerY(y,h) <= 285) and ((centerY(y,h) >= 270))) and ((centerX(x,w) >= 230) and ((centerY(x,w) < 295))):
				if((framecount-lastframeleftright)>=15):
					if (h>70):
						truck+=1
						lastframeleftright = framecount
						down += 1
					elif (h>45):
						minibus+=1
						lastframeleftright = framecount
						down += 1
					else:
						carcount += 1
						lastframeleftright = framecount
						down+=1

			#right lane left counter
			if ((centerY(y,h) <= 285) and ((centerY(y,h) >= 270))) and ((centerX(x,w) > 385) and ((centerY(x,w) < 450))):
				if((framecount-lastframerightleft)>=15):
					if (h>70):
						truck+=1
						lastframerightleft = framecount
						up += 1
					elif (h>45):
						minibus+=1
						lastframerightleft = framecount
						up += 1
					else:
						carcount += 1
						lastframerightleft = framecount
						up+=1
				
			#right lane right counter
			if ((centerY(y,h) <= 285) and ((centerY(y,h) >= 270))) and ((centerX(x,w) >= 450) and ((centerY(x,w) < 600))):
				if((framecount-lastframerightright)>=15):
					if (h>70):
						truck+=1
						lastframerightright = framecount
						up += 1
					elif (h>45):
						minibus+=1
						lastframerightright = framecount
						up += 1
					else:
						carcount += 1
						lastframerightright = framecount
						up+=1

	#texts=================================================================================================================
	#frame
	cv2.putText(frame,"Frame: {0}".format(framecount) , (530, 50), font, 0.5, (0,0,0), 2, cv2.LINE_AA)
	#going up cars
	cv2.putText(frame,"Up: {0}".format(up) , (550, 320), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
	#goind down cars
	cv2.putText(frame,"Down: {0}".format(down) , (550, 340), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
	
	#total cars
	cv2.putText(frame,"Truck: {0}".format(truck) , (20, 300), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
	#minibuss
	cv2.putText(frame,"Minibus: {0}".format(minibus) , (20, 320), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
	#goind down cars
	cv2.putText(frame,"Car: {0}".format(carcount) , (20, 340), font, 0.5, (0,0,255), 2, cv2.LINE_AA)

	#shows=============================================================================================================
	#just showing the masks and operations

	#showing the result
	cv2.imshow('frame', frame)

	#exit
	k = cv2.waitKey(30) & 0xFF
	if k == 27:
		break;



cv2.destroyAllWindows()
cap.release()