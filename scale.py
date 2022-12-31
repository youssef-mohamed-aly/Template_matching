import numpy as np
import cv2 as cv

img = cv.resize(cv.imread("assets/soccer_practice.jpg", 0), (0, 0), fx=0.8, fy=0.8)

template = cv.resize(cv.imread("assets/ball.PNG", 0), (0, 0), fx=0.8, fy=0.8)

h, w = template.shape

method = cv.TM_CCOEFF_NORMED

img2 = img.copy()

result = cv.matchTemplate(img2, template, method)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

location = max_loc

bottom_right = (location[0] + w, location[1] + h)  

cv.rectangle(img2, location, bottom_right, 255, 5)

cv.imshow('Match', img2)

cv.waitKey(0)

cv.destroyAllWindows()