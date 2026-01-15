import torch
# Импортируем нужный класс, чтобы PyTorch знал, как его обрабатывать
from ultralytics.nn.tasks import DetectionModel 

# 1. Загружаем чекпоинт с отключенным фильтром weights_only
ckpt = torch.load('rail_project/mono_mvp_v24/weights/best.pt', map_location='cpu', weights_only=False)
model = ckpt['model']

# 2. Извлекаем веса первого слоя
# В YOLOv8 это слой 0, Conv
old_weight = model.model[0].conv.weight.data # Было [16, 3, 3, 3]
print(f"Исходные веса: {old_weight.shape}")

# 3. Схлопываем 3 канала в 1 (берем среднее значение)
new_weight = old_weight.mean(dim=1, keepdim=True) # Стало [16, 1, 3, 3]
print(f"Новые веса: {new_weight.shape}")

# 4. Перезаписываем данные в модели
model.model[0].conv.weight.data = new_weight
model.model[0].conv.in_channels = 1

# 5. Сохраняем обновленный чекпоинт
torch.save(ckpt, 'rail_project/mono_mvp_v24/weights/best_mono.pt')
print("Успешно! Создан файл: rail_project/mono_mvp_v24/weights/best_mono.pt")
