import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

# Конфигурация
MODEL_PATH = 'rail_project/mono_mvp_v24/weights/best_mono.pt'
IMAGE_PATH = 'dataset/test/images/1_blur_jpg.rf.1e18e73e480e169d199fac3dc219ba6c.jpg'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_benchmark():
    # 1. Загрузка модели и перенос на GPU
    print(f"Загрузка модели на {DEVICE}...")
    model = YOLO(MODEL_PATH, task='detect')
    model.model.to(DEVICE)
    model.model.eval()

    # 2. Подготовка тестового тензора (1 канал, 256x512)
    # Создаем тензор напрямую на GPU, чтобы исключить влияние шины PCI-E при замере
    input_tensor = torch.zeros((1, 1, 256, 512)).to(DEVICE).float()

    # 3. ПРОГРЕВ (Warmup) - крайне важно для точных цифр
    print("Прогрев видеокарты...")
    for _ in range(50):
        with torch.no_grad():
            _ = model.model(input_tensor)
    
    # 4. ЗАМЕР ЧИСТОГО ИНФЕРЕНСА (Чистая математика GPU)
    print("Запуск бенчмарка (1000 итераций)...")
    torch.cuda.synchronize() # Ждем завершения всех фоновых задач GPU
    t_start = time.time()
    
    with torch.no_grad():
        for _ in range(1000):
            _ = model.model(input_tensor)
            
    torch.cuda.synchronize() # Ждем завершения вычислений
    t_end = time.time()

    avg_latency = ((t_end - t_start) / 1000) * 1000
    fps = 1000 / avg_latency

    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТЫ (RTX 4060TI):")
    print(f"Средняя задержка (Latency): {avg_latency:.3f} мс")
    print("="*30 + "\n")

    # 5. ДЕМОНСТРАЦИЯ НА РЕАЛЬНОМ ИЗОБРАЖЕНИИ
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None:
        print("Изображение для теста не найдено.")
        return

    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (512, 256))
    
    # Подготовка реального кадра
    frame_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.no_grad():
        preds = model.model(frame_tensor)
        # Применяем NMS
        results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    # Отрисовка
    det = results[0]
    if len(det) > 0:
        h_ratio = orig_img.shape[0] / 256
        w_ratio = orig_img.shape[1] / 512
        
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, [xyxy[0]*w_ratio, xyxy[1]*h_ratio, xyxy[2]*w_ratio, xyxy[3]*h_ratio])
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(orig_img, f"Defect {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Обнаружен дефект: {conf:.2f}")

    cv2.imshow("Final Test RTX 4060TI", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_benchmark()
