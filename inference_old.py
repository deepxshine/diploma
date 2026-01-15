import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
# Универсальный импорт NMS
from ultralytics.utils.nms import non_max_suppression

# 1. Загружаем модель
# Используем нашу пропатченную mono-модель

device = torch.device('cuda')
model = YOLO('rail_project/mono_mvp_v24/weights/best_mono.pt', task='detect')
model.model.to(device)
model.model.eval() 
def predict_mono(img_path):
    # Читаем картинку
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Ошибка: Файл {img_path} не найден")
        return
    
    # Конвертация в Grayscale (1 канал)
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Подготовка тензора [1, 1, 256, 512]
    # OpenCV resize: (width, height) -> (512, 256)
    img_resized = cv2.resize(gray_img, (512, 256))
    input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
    input_tensor /= 255.0
    input_tensor = input_tensor.to(device)
    # 2. Замер чистого инференса (для Главы 5 диплома)
    inference_times = 0
    for _ in range(100):
        t1 = time.time()
        with torch.no_grad():
            preds = model.model(input_tensor)
        t2 = time.time()
    
        inference_time = (t2 - t1) * 1000
        inference_times += inference_time
        fps = 1 / (t2 - t1)

    print(f"Inference: {inference_times/100:.2f} ms | Est. FPS: {fps:.0f}")

    # 3. Пост-процессинг (NMS)
    # Если импорт выше не сработал, можно вызвать через: 
    # from ultralytics.utils.ops import non_max_suppression
    results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    # 4. Отрисовка
    if len(results[0]) > 0:
        print(f"Найдено дефектов: {len(results[0])}")
        
        h_ratio = orig_img.shape[0] / 256
        w_ratio = orig_img.shape[1] / 512
        
        for det in results[0]:
            x1, y1, x2, y2, conf, cls = det
            x1, x2 = int(x1 * w_ratio), int(x2 * w_ratio)
            y1, y2 = int(y1 * h_ratio), int(y2 * h_ratio)

            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(orig_img, f"Defect {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        print("Дефектов не обнаружено")
    
    cv2.imshow("Result", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Убедитесь, что путь к фото верный
    test_img = 'dataset/test/images/1_blur_jpg.rf.1e18e73e480e169d199fac3dc219ba6c.jpg'
    predict_mono(test_img)
