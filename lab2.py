import numpy as np
import cv2 as cv

imageSource = '/home/ekolosova/Desktop/bear_little.jpg'
image = cv.imread(imageSource)
if image is not None:
    cv.imshow('Original image', image)
elif image is None:
    print("Error loading image")


def create_kernel():
    kernel = np.ones((5, 5, 3))
    print(kernel.shape)
    kernel /= kernel.shape[0] * kernel.shape[1] * kernel.shape[2]
    return kernel

kernel = create_kernel()
print(kernel.dtype)

width, height, k = image.shape

r = int(kernel.shape[1]/2)

res = np.zeros((width, height, 1), np.uint8)

for i in range(width):
    for j in range(height):
        for ka in range(k):
            for w in range(-r, r):
                for h in range(-r, r):
                    i_new = np.clip(i + w, 0, image.shape[0] - 1)
                    j_new = np.clip(j + h, 0, image.shape[1] - 1)
                    res[i][j] += int(kernel[r+w][r+h][ka] * image[i_new][j_new][ka])

cv.imshow('Result', res)

k = cv.waitKey(0)
cv.destroyAllWindows()