import cv2
import matplotlib.pyplot as plt
import numpy as np

fileDir = 'imageDatabase/'
objFile = 'JonquilFlowers.jpg'

img = cv2.imread(fileDir + objFile, 0)
# img2 = plt.imread(fileDir + objFile)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.subplot(131)
plt.title('cv2 hist')
plt.plot(histr)

plt.subplot(132)
plt.title('matplotlib hist')
plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(133)
plt.title('matplotlib bar')
x = np.arange(4)
histr = np.ravel(histr)
size = np.arange(histr.size)
plt.bar(size, histr)

plt.show()
