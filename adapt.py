import torch
import os

# Импортируем классы Ultralytics, чтобы PyTorch знал их при загрузке
try:
	from ultralytics.nn.tasks import DetectionModel
	from ultralytics.nn.modules.conv import Conv

	# Добавляем их в безопасный список (рекомендуемый способ для PyTorch 2.6+)
	torch.serialization.add_safe_globals([DetectionModel, Conv])
except ImportError:
	pass


def adapt_model_to_gray(model_path='yolov8n.pt'):
	if not os.path.exists(model_path):
		print(f"Ошибка: Файл {model_path} не найден!")
		return

	print(f"Загрузка локального файла: {model_path}...")

	# Используем weights_only=False, так как файл содержит кастомные объекты YOLO
	ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

	# Проверяем структуру (в YOLOv8 модель лежит в ключе 'model')
	if 'model' in ckpt:
		model_state = ckpt['model'].state_dict()
		layer_name = 'model.0.conv.weight'

		if layer_name in model_state:
			w_rgb = model_state[layer_name]
			print(f"Исходные веса (RGB): {w_rgb.shape}")

			# Усреднение каналов по формуле (4.2) из вашего ТЗ
			w_mono = w_rgb.mean(dim=1, keepdim=True)

			model_state[layer_name] = w_mono
			print(f"Новые веса (Mono): {w_mono.shape}")

			save_path = 'yolov8n_mono.pt'
			torch.save(ckpt, save_path)
			print(f"--- Успешно сохранено в {save_path} ---")
		else:
			print(
				f"Ключ {layer_name} не найден. Доступные ключи: {list(model_state.keys())[:5]}"
				)
	else:
		print("Ошибка: Ключ 'model' не найден в чекпоинте.")


if __name__ == "__main__":
	adapt_model_to_gray()