import sys

sys.path.append("../src")

import numpy as np
import cv2
import math
import fractions

scaleFactor = 1.5
fraction = fractions.Fraction(scaleFactor).limit_denominator()

def resize(original_img, newHeight, newWidth):
    old_h, old_w, c = original_img.shape
    resized = np.zeros((newHeight, newWidth, c))
    w_scale_factor = old_w / newWidth if newHeight != 0 else 0
    h_scale_factor = old_h / newHeight if newWidth != 0 else 0

    for i in range(newHeight):
        for j in range(newWidth):
            x = i * h_scale_factor
            y = j * w_scale_factor
            x_floor = math.floor(x)
            y_floor = math.floor(y)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_ceil = min(old_w - 1, math.ceil(y))

            if x_ceil == x_floor and y_ceil == y_floor:
                resized[i, j, :] = original_img[x_floor, y_floor, :]
            elif x_ceil == x_floor:
                q1 = original_img[x_floor, y_floor, :]
                q2 = original_img[x_floor, y_ceil, :]
                resized[i, j, :] = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif y_ceil == y_floor:
                q1 = original_img[x_floor, y_floor, :]
                q2 = original_img[x_ceil, y_floor, :]
                resized[i, j, :] = q1 * (x_ceil - x) + q2 * (x - x_floor)
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                resized[i, j, :] = q1 * (y_ceil - y) + q2 * (y - y_floor)
    return resized

image = cv2.imread("milky-way.jpg")
heigh, width, f = image.shape
finalHeight, finalWidth = int(heigh * scaleFactor), int(width * scaleFactor)
print(finalWidth, finalHeight)

resizedImage = resize(image, finalHeight, finalWidth)

cv2.imwrite(f"resizedby_{scaleFactor}.png", resizedImage)
