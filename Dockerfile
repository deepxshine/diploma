FROM nvcr.io/nvidia/deepstream:6.3-samples

# Установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-gstreamer-1.0 \
    libgirepository1.0-dev \
    pkg-config \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Python пакетов
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python-headless \
    opencv-contrib-python-headless \
    pycairo \
    PyGObject

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем основной скрипт и папку с изображениями
COPY main.py .
COPY augmented/ ./augmented/

# Устанавливаем переменные окружения
ENV DISPLAY=:99
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Запускаем Xvfb в фоне (для headless режима)
RUN apt-get update && apt-get install -y xvfb && rm -rf /var/lib/apt/lists/*
RUN Xvfb :99 -screen 0 1920x1080x24 &

# Команда для запуска
CMD ["python3", "main.py"]