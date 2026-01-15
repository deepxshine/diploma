from ultralytics import YOLO

# Загружаем модель с нашей структурой и адаптированными весами
model = YOLO('yolov8-rail.yaml').load('yolov8n_mono.pt')

# Запуск обучения
results = model.train(
    data='rail_data.yaml',
    epochs=100,
    imgsz=(512, 256), # Ваша размерность
    batch=16,
    augment=True,
    mosaic=1.0,      # Помогает при малом датасете
    freeze=10,       # Замораживаем первые 10 слоев (бэкбон)
    device=0         # GPU
)

model.export(format='engine', device=0, half=True, simplify=True)