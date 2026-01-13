import cv2
import os
import numpy as np

input_dir = "output"
aug_dir = "augmented"
os.makedirs(aug_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gamma_correction(img, gamma=1.0):
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, lut)

def add_gaussian_noise(img, sigma=2):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def gaussian_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0.5)

# Параметры аугментаций
shifts = [-4, -2, 2, 4]
alphas = [0.9, 1.1]
betas = [-10, 10]
gammas = [0.9, 1.1]

for fname in files:
    path = os.path.join(input_dir, fname)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    name, ext = os.path.splitext(fname)

    # Сохраняем оригинал
    cv2.imwrite(os.path.join(aug_dir, f"{name}_orig{ext}"), img)

    # 1. Сдвиги
    for dx in shifts:
        for dy in shifts:
            shifted = shift(img, dx, dy)
            cv2.imwrite(os.path.join(aug_dir, f"{name}_shift_{dx}_{dy}{ext}"), shifted)

    # 2. Яркость / контраст
    for alpha in alphas:
        for beta in betas:
            bc = adjust_brightness_contrast(img, alpha=alpha, beta=beta)
            cv2.imwrite(os.path.join(aug_dir, f"{name}_bc_{alpha}_{beta}{ext}"), bc)

    # 3. Гамма
    for g in gammas:
        g_img = gamma_correction(img, gamma=g)
        cv2.imwrite(os.path.join(aug_dir, f"{name}_gamma_{g}{ext}"), g_img)

    # 4. Лёгкое размытие
    blurred = gaussian_blur(img)
    cv2.imwrite(os.path.join(aug_dir, f"{name}_blur{ext}"), blurred)

    # 5. Шум
    noisy = add_gaussian_noise(img)
    cv2.imwrite(os.path.join(aug_dir, f"{name}_noise{ext}"), noisy)

    # 6. Комбинации: shift + gamma + noise
    for dx in shifts:
        for dy in shifts:
            shifted = shift(img, dx, dy)
            for g in gammas:
                sg = gamma_correction(shifted, gamma=g)
                sn = add_gaussian_noise(sg)
                cv2.imwrite(os.path.join(aug_dir, f"{name}_s{dx}_{dy}_g{g}_n{ext}"), sn)
