FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch


# Установка зависимостей Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-numpy \
    python3-opencv \
    libgirepository1.0-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Python модулей
RUN pip3 install --upgrade pip && \
    pip3 install opencv-python opencv-python-headless PyGObject

# Установка GStreamer Python биндингов
RUN apt-get update && apt-get install -y \
    python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование исходного кода
COPY main.py main.py

# Установка прав
RUN chmod +x /app/main.py

# Команда запуска
CMD ["python3", "main.py"]
