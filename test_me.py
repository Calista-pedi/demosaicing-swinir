import cv2
import numpy as np
import matplotlib.pyplot as plt

# 原图
clean = cv2.imread('testsets/McMaster/2.tif')

# 加噪
noise = np.random.normal(20, 25, clean.shape)
noisy = clean + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# 模型结果
denoised = cv2.imread('results/swinir_color_dn_noise15/2_SwinIR.png')

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Clean"); plt.imshow(clean[:,:,::-1])
plt.subplot(1,3,3); plt.title("Noisy"); plt.imshow(noisy[:,:,::-1])
plt.subplot(1,3,2); plt.title("Denoised"); plt.imshow(denoised[:,:,::-1])
plt.show()