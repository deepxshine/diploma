import cv2
import os
import numpy as np
import random

input_dir = 'wd_prepared'
output_dir = 'wd_augmented'

if not os.path.exists(output_dir):
	os.makedirs(output_dir)


def add_noise(image):
	# Добавление случайного шума
	gauss = np.random.normal(0, 0.1, image.size)
	gauss = gauss.reshape(image.shape[0], image.shape[1]).astype('uint8')
	return cv2.add(image, gauss)


def change_brightness(image, value):
	# Изменение яркости
	hsv = cv2.cvtColor(
		image, cv2.COLOR_GRAY2BGR
		)  # временный перевод для корректной работы
	hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	v = cv2.add(v, value)
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Список файлов
files = [f for f in os.listdir(input_dir) if
		 f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in files:
	img_path = os.path.join(input_dir, filename)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Читаем сразу в ЧБ

	name = os.path.splitext(filename)[0]

	# Список трансформаций (всего 10 вариантов для каждого фото)
	variations = [
		("orig", img),
		("flip", cv2.flip(img, 1)),  # Горизонтальное отражение
		("bright", cv2.convertScaleAbs(img, alpha=1.0, beta=30)),  # Ярче
		("dark", cv2.convertScaleAbs(img, alpha=1.0, beta=-30)),  # Темнее
		("contrast", cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
		# Контрастнее
		("blur", cv2.GaussianBlur(img, (5, 5), 0)),  # Размытие
		("noise", add_noise(img)),  # Шум
		("flip_bright",
		 cv2.flip(cv2.convertScaleAbs(img, alpha=1.0, beta=20), 1)),
		# Отражение + яркость
		("flip_blur", cv2.flip(cv2.medianBlur(img, 5), 1)),
		# Отражение + размытие
		("sharp",
		 cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0))
		# Резкость
	]

	for suffix, processed_img in variations:
		new_name = f"{name}_{suffix}.jpg"
		cv2.imwrite(os.path.join(output_dir, new_name), processed_img)

print(
	f"Готово! В папке {output_dir} теперь {len(os.listdir(output_dir))} изображений."
	)