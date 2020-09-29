import cv2
import numpy as np

img = cv2.imread('base64.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255,cv2.THRESH_BINARY_INV)
resultado = np.hstack([binI])
cv2.imwrite('base64CV2.png',resultado)