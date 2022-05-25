# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:53:44 2021

@author: Ayan
"""

import cv2 as cv

def rescaleFrame(frame, scale=0.75):    # function to resize the image
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    
    dimensions=(width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('picture5.jpg')

resized_image = rescaleFrame(img)

cv.imshow('Face', resized_image)

cv.waitKey(0)

gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)



haar_cascade = cv.CascadeClassifier(r'C:\Users\Ayan\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1 )

print(f'Number of faces found = {len(faces_react)}')

for(x,y,w,h) in faces_react:
    cv.rectangle(resized_image, (x,y), (x+w,y+h), (0,255,0), thickness=2) 

cv.imshow('Detected Faces', resized_image)

cv.waitKey(0)
