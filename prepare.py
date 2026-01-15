import os
import random
import shutil

# исходная и целевая директории
src = "dataset_old/normal"
dst = "dataset/train/images"

# создаём целевую папку, если нет
os.makedirs(dst, exist_ok=True)

# получаем список только файлов (без подкаталогов)
files = [
    f for f in os.listdir(src)
    if os.path.isfile(os.path.join(src, f))
]

# если файлов меньше 50, берём все
k = min(50, len(files))

# выбираем 50 случайных файлов
selected = random.sample(files, k)

# переносим
for name in selected:
    src_path = os.path.join(src, name)
    dst_path = os.path.join(dst, name)
    shutil.move(src_path, dst_path)

print(f"Перемещено файлов: {k}")
