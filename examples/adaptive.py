import cv2
import torch
import fastcv
import numpy as np


img = cv2.imread("artifacts/test.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img)

max_value = 255.0
block_size = 15
C = 5.0

func = fastcv.adaptive_threshold
mean_result = func(img_tensor, max_value, 0, 0, block_size, C)
gauss_result = func(img_tensor, max_value, 1, 0, block_size, C)

cv2.imwrite("output_mean.jpg", mean_result.numpy())
cv2.imwrite("output_gauss.jpg", gauss_result.numpy())

print("saved adaptive treshold mean and gauss images.")
