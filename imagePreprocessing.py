import cv2
import numpy as np


img = cv2.imread('dataset/train/1/1.png')
img_bw = 255*(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)>5).astype('uint8')
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
mask = np.dstack([mask,mask,mask])/255
out = img*mask
#cv2.imshow('Output', out)


# image = cv2.imread('input1.jpg')
#img_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
img_gray = out
img_gray = cv2.medianBlur(img_gray, 5)
edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
ret,mask =cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)
image2 = cv2.bitwise_and(image, image, mask=mask)
image2 = cv2.medianBlur(image2, 3)  # this
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imwrite('output.png', mask)
