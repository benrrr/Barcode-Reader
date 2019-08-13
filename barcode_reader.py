# The following submission is a
# barcode and QR code detection 
# script that can also parse
# some barcode data. The script
# was researched and developed by:
#
# 	Robert Vaughan - C15341261
# 		Researched detection
# 	Ben Ryan - C15507277
# 		Researched reading the data
# 	Mohamad Zabad - C15745405
# 		Researched image transformation
#
# All research and explanations of how each person came to their 
# solution over the number of weeks can be found within the project
# blogs for as stated in class, no explanation within the code is required.

import numpy as np
import cv2
import math
import easygui
 
def intCheck(string):
    try: 
        int(string)
        return True
    except ValueError:
        return False
		
# Takes a an image and 
# finds the std and ceils it
# to the nearest odd number
def nearestOddInteger(val):
	return int(np.ceil(np.std(val)) // 2 * 2 + 1)

# Checks to see if an image is
# a barcode or not
def barcodeCheck(img):
	height, width, _ = np.shape(img)

	if width > (height + 5) or width < (height - 5):
		print("Barcode: " + str(width) + " " + str(height))
		return True
	else:
		print("QR: " + str(width) + " " + str(height))
		return False

# Code to detect if a barcode
def codeDetection(img): 
	image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width, _ = np.shape(img)
	#print(np.shape(img))
	
	# Creating a blur kernal
	blur_k = (5,5)

	image_gray = cv2.GaussianBlur(image_gray, blur_k, 0)

	# Canny's the image with a dynamic threshold value
	canny = cv2.Canny(image_gray, threshold1=255-nearestOddInteger(image_gray), threshold2=255)
	
	# Using a percentage of the square area to draw our kernal
	area = int((math.sqrt(width * height)) * .02)
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (area,area))

	# Dilate the image to bind together areas with heavy
	# positive pixels
	dilation = cv2.dilate(canny, structEl, iterations = 2)

	# Takes the boundary from the contoured areas
	# To ensure that solid structures are present
	structK = (4,4)
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, structK)
	boundary = cv2.morphologyEx(dilation,cv2.MORPH_GRADIENT,structEl)

	# Getting the contours of the dilated image
	contours, _ = cv2.findContours(boundary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(image_gray, contours, -1, 255, 3)

	# Finding the biggest contour, which 
	# should be our code
	c = max(contours, key = cv2.contourArea)

	# Gets the bounding co-ordinates of the
	# biggest contour to get the image
	x, y, width, height = cv2.boundingRect(c)

	# Get the rotated rect.
	# This is for later use
	# for the rotation stages of the image
	rotRect = cv2.minAreaRect(c)

	# Creates a copy of an image
	# and draws an outline of the detected area from
	# our contouring
	img_copy = img.copy()
	cv2.rectangle(img_copy,(x, y),(x + width, y + height), (0,255,0), 2)
	
	crop_img = img[y:y + height, x:x + width]
 
	return crop_img, rotRect, img_copy
		
#handles decoding barcodes
def decodeBarcode(img):	
	#get dimensions
	h, w, c = np.shape(img)
	
	#convert to grey and threshold it
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
	
	#closing image to clean up the barcode, remove numbers so they don't interfere with decoding
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h/4)))
	morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structEl)
	
	#if previous morph left mostly white, barcode is probably rotated 90 degrees
	#if this is the case morph using different shape structuring element
	if np.mean(morphed) > 240:
		structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w/4), 1))
		morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structEl)
		
	#get the binary code for each section of the barcode
	#black bar 1 white bar 0
	binaryCode, finalImg = getBinary(morphed, h, w)
	
	#convert the binary code found in to the acutal barcode numbers
	finalCode = convertBinary(binaryCode)

	if finalCode is None:		
		finalCode = "Could not read correctly"
	
	print("CODE: " + finalCode)	
	return finalCode
	
#gets a binary string from the image
def getBinary(img, h, w):
	#calculate the exact boundaries/characteristics of the barcode for decoding		
	startX, endX, startY, endY, barWidth, img = findBounds(img, h, w)

	binaryCode = ''
	
	#iterate over the barcode using above variables
	for column in range(startX, endX, barWidth):
		#select bar
		line = img[startY: endY, column:column+barWidth]
		
		#find the avg value for the pixels in the selected bar
		
		avg = np.mean(line)

		#if in the upper half of values binary 0, else binary 1
		if avg > int((255/2)):
			binaryCode += '0'
		else:
			binaryCode += '1'
			
	return binaryCode, img


def rotated(img_in):

	# copy of the image used for the rotation in the end
	img_copy = img_in.copy()

	# Grayscale converting
	image_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
	blur_k = (5,5)
	#Applying a GaussianBlur
	image_gray = cv2.GaussianBlur(image_gray, blur_k, 0)
	#Reversing the colours
	gray = cv2.bitwise_not(image_gray)
	# threshold the image
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# Storing the positive pixel locations ( 0 is off and 1 is on )
	coords = np.column_stack(np.where(thresh > 0))
	# minAreaRect will determine the angle needed to align everything
	angle = cv2.minAreaRect(coords)[-1]


	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)

	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle

	# rotating the image
	(h, w) = thresh.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(img_copy, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	return rotated


#finding exact boundaries to be read
def findBounds(img, h , w):
	startX = 0 
	endX = 0
	startY = 0 
	endY = 0
	barWidth = 1
	
	#calculate startx and barWidth
	#iterating over columns finding mean values until it finds one that is on avg black
	for x in range(0,w):
		#cut the image in single pixel wide, full height bars, across the image
		column = img[0:h,x:x+1]
			
		#if the mean is below 128 assume black, mark as start, and start recording the width of bars
		if np.mean(column) < 230:
			startX = x
			
			#keep iterating from the start counting how wide the bar is
			for x2 in range(x,w):
				column2 = img[0:h,x2+1]

				#once it goes back to white on avg break, stop counting bar width
				if np.mean(column2) > 230:
					break
				else:
					barWidth +=1
			break
		
	#calc endX, working backwwards from width to 0 finding where the end of the barcode is
	for x in range(w-1, 0, -1):
		column = img[0:h, x:x+1]
		
		#when found save it and break
		if np.mean(column) < 230:
			endX = x
			break
			
	#calc start y as above
	for y in range(0, h):
		row = img[y:y+1, startX:endX]
		
		if np.mean(row) < 175:
			startY = y
			break;
			
	#calc end y as above
	for y in range(h-1, 0, -1):
		row = img[y:y+1,startX:endX]
	
		if np.mean(row) < 175:
			endY = y
			break;

	#if not oriented correctly rotate 90 degress
	if endX - startX < endY - startY:
		#crop the found area
		crop_img = img[startY:endY, startX:endX]
		h, w = np.shape(crop_img)
		
		#create new white image to fit the cropped image so rotating doesn't lose any code
		wh = int(math.hypot(w, -h))
		newImg = np.zeros((wh,wh), np.uint8)
		newImg.fill(255)
		
		x1 = int((wh-w)/2)
		y1 = int((wh-h)/2)
		
		newImg[y1:y1+h, x1:x1+w] = crop_img
		
		c = (wh/2,wh/2)
		
		#rotate 90
		rotMx = cv2.getRotationMatrix2D(c, 90, 1)
		rotImg = cv2.warpAffine(newImg, rotMx, (wh,wh), borderValue = (255,255,255))

		#find bounds in new image and overwrite previous bounds
		h,w = np.shape(rotImg)
		startX, endX, startY, endY, barWidth, img = findBounds(rotImg, h , w)
			
	showImage("Rot", img)
	return startX, endX, startY, endY, barWidth, img
	
#converts from binary to decimal
def convertBinary(binaryCode):
	#UPC-A codes
	left = ['0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011']
	right = ['1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100']

	#if error found
	if errorCheck(binaryCode) is not True:
		return
		
	#if the guard bars are in the correct location then the following should split the left and right side 
	#exactly for decodoing
	leftSide = binaryCode[3:45]
	rightSide = binaryCode[50:92]
	
	finalLeft = convertSide(leftSide, left)
	finalRight = convertSide(rightSide, right)
	
	#if converted sides are empty, code is probably upside down, so flip it
	if finalLeft == '' or finalRight == '':
		rightSide, leftSide = leftSide[::-1], rightSide[::-1]
		
		finalLeft = convertSide(leftSide, left)
		finalRight = convertSide(rightSide, right)
		
	finalCode = finalLeft + finalRight
	return finalCode
	
#checks guard bars are in the correct place and code is correct length
def errorCheck(binaryCode):
	sideGuard = '101'
	midGuard = '01010'

	print(len(binaryCode), binaryCode)
	if len(binaryCode) != 95:
			print("incorrect length found")
			return False
			
	#checking if guard bars are in the correct place, otherwise don't bother checking the rest
	#only works for perfectly aligned and read.
	if binaryCode[0:3] == sideGuard and binaryCode[45:50] == midGuard and  binaryCode[92:95] == sideGuard:
		return True
	
	return False
	
#converts a side of the barcode from binary to decimal using provided list
def convertSide(side, bin):
	final = ''
	
	for section in range(0,len(side)+1, 7):
		for code in bin:
			if side[section:section+7] == code:
				final += str(bin.index(code))
				break

	return final
 
#shows image and hold window open
def showImage(title, image):
	s = np.shape(image)

	if s[0] > 1000 or s[1] > 1000:
		image = cv2.resize(image, (int(s[1]*.5), int(s[0]*.5)))
	
	cv2.imshow(title, image)
	cv2.waitKey(0)
	
#aligns the image based on the rotation in the provided rotated rectangle
def align(crop_img, rotRect):
	h, w, c = np.shape(crop_img)

	#create new white image for aligning
	wh = int(math.hypot(w, - h))
	newImg = np.zeros((wh, wh, c), np.uint8)
	newImg.fill(255)

	x1 = int((wh-w)/2)
	y1 = int((wh-h)/2)

	# put orig inside new img
	newImg[y1:y1+h, x1:x1+w] = crop_img

	# get centre
	c = (wh/2,wh/2)

	#rotate
	rotMx = cv2.getRotationMatrix2D(c, rotRect[2], 1)
	rotImg = cv2.warpAffine(newImg, rotMx, (wh,wh), borderValue = (255,255,255))

	return rotImg

def main():
	filename = easygui.fileopenbox()
	img = cv2.imread(filename)

	copy = img.copy()
	img = rotated(img)
	img, rotRect, drawn = codeDetection(img)

	finalCode = 'QR Code'
	if barcodeCheck(img):
		aligned = align(img, rotRect)
		finalCode = decodeBarcode(aligned)
		
	showImage(finalCode, drawn)

if __name__ == "__main__":
    main()