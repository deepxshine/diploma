import os

img_dir = "dataset/train/images"
lbl_dir = "dataset/train/labels"

images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
labels = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

print(f"Найдено изображений: {len(images)}")
print(f"Найдено файлов разметки (.txt): {len(labels)}")

# Проверка соответствия первого файла
if len(images) > 0:
    base_name = os.path.splitext(images[0])[0]
    if os.path.exists(os.path.join(lbl_dir, base_name + ".txt")):
        print(f"Связь подтверждена: {images[0]} <--> {base_name}.txt")
    else:
        print(f"ОШИБКА: Для {images[0]} не найден файл разметки в {lbl_dir}")