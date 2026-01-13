import sys
import time
import threading
import numpy as np
import cv2
import gi
import os
import glob
import subprocess

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# --- КОНФИГУРАЦИЯ ---
WIDTH = 1920
HEIGHT = 1080
FPS = 30
IMAGE_FOLDER = "augmented"
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

# Проверяем, находимся ли мы в Docker
IS_DOCKER = os.path.exists('/.dockerenv')

# Для статистики
frame_count = 0
start_time = time.time()
pipeline = None


class RingBufferFeeder:
    def __init__(self, image_folder, target_w, target_h):
        self.frames_buffer = []
        self.current_idx = 0
        self.lock = threading.Lock()
        
        image_files = self.get_image_files(image_folder)
        
        if not image_files:
            print(f"⚠️ Папка '{image_folder}' пуста. Создаю тестовые изображения...")
            self.create_test_images(image_folder)
            image_files = self.get_image_files(image_folder)
        
        print(f"[RAM] Найдено {len(image_files)} изображений")
        
        for idx, fp in enumerate(image_files):
            img = cv2.imread(fp)
            if img is None:
                print(f"⚠️ Ошибка чтения {fp}")
                continue
            
            img_resized = self.letterbox_resize(img, target_w, target_h)
            img_rgba = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA)
            
            self.frames_buffer.append({
                'bytes': img_rgba.tobytes(),
                'filename': os.path.basename(fp)
            })
        
        if not self.frames_buffer:
            raise Exception("Не удалось загрузить ни одного изображения!")
        
        print(f"[RAM] Загружено {len(self.frames_buffer)} кадров")
    
    def get_image_files(self, folder):
        """Получаем список изображений"""
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(folder, ext)
            image_files.extend(glob.glob(pattern, recursive=False))
        image_files.sort()
        return image_files
    
    def create_test_images(self, folder, num=10):
        """Создание тестовых изображений если папка пуста"""
        os.makedirs(folder, exist_ok=True)
        
        for i in range(num):
            img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            color = (i * 25, i * 50, i * 75)
            img[:] = color
            
            cv2.putText(img, f"Docker Test {i+1}", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(img, f"{WIDTH}x{HEIGHT}", (100, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            filename = os.path.join(folder, f"test_{i+1:03d}.jpg")
            cv2.imwrite(filename, img)
            print(f"  Создано: {filename}")
    
    def letterbox_resize(self, image, target_w, target_h):
        """Letterbox ресайз"""
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def get_next_frame_bytes(self):
        with self.lock:
            data = self.frames_buffer[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.frames_buffer)
            return data['bytes']


feeder = None


def need_data(src, length):
    global feeder, frame_count
    
    try:
        data = feeder.get_next_frame_bytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        
        # Таймстемпы
        buf.pts = frame_count * Gst.SECOND // FPS
        buf.duration = Gst.SECOND // FPS
        
        retval = src.emit('push-buffer', buf)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[STATS] Frames: {frame_count}, FPS: {fps:.1f}")
            
    except Exception as e:
        print(f"Error in need_data: {e}")


def bus_call(bus, message, loop):
    t = message.type
    
    if t == Gst.MessageType.EOS:
        print("[Pipeline] End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[Pipeline] Error: {err}")
        print(f"[Pipeline] Debug: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[Pipeline] Warning: {err}")
    
    return True


def create_pipeline():
    """Создание пайплайна с учетом Docker окружения"""
    global pipeline
    
    print("\n" + "=" * 60)
    print("Создание пайплайна")
    print(f"Docker mode: {IS_DOCKER}")
    print("=" * 60)
    
    pipeline = Gst.Pipeline()
    
    # 1. AppSrc
    appsrc = Gst.ElementFactory.make("appsrc", "source")
    caps = Gst.Caps.from_string(
        f"video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate={FPS}/1"
    )
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("do-timestamp", False)
    appsrc.set_property("is-live", True)
    
    # 2. nvvideoconvert
    nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "uploader")
    
    # 3. nvstreammux
    streammux = Gst.ElementFactory.make("nvstreammux", "muxer")
    streammux.set_property('width', WIDTH)
    streammux.set_property('height', HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    
    # 4. nvvideoconvert
    nvvidconv_out = Gst.ElementFactory.make("nvvideoconvert", "converter")
    
    # 5. nvosd
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
    nvosd.set_property('display-text', True)
    
    # 6. Sink - разный в зависимости от окружения
    if IS_DOCKER:
        print("  Использую fakesink для Docker")
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("qos", False)
    else:
        print("  Использую nveglglessink для локального запуска")
        sink = Gst.ElementFactory.make("nveglglessink", "sink")
        sink.set_property("sync", False)
    
    # Добавляем элементы
    elements = [appsrc, nvvidconv_src, streammux, nvvidconv_out, nvosd, sink]
    
    for element in elements:
        if element:
            pipeline.add(element)
            print(f"  ✓ {element.get_factory().get_name()}")
    
    # Линковка
    print("\nЛинковка элементов...")
    
    # appsrc -> nvvidconv_src
    if not appsrc.link(nvvidconv_src):
        print("❌ appsrc -> nvvidconv_src")
        return None
    
    # nvvidconv_src -> streammux
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = nvvidconv_src.get_static_pad("src")
    if srcpad and sinkpad:
        srcpad.link(sinkpad)
    else:
        print("❌ nvvidconv_src -> streammux")
        return None
    
    # streammux -> nvvidconv_out -> nvosd -> sink
    if not streammux.link(nvvidconv_out):
        print("❌ streammux -> nvvidconv_out")
        return None
    
    if not nvvidconv_out.link(nvosd):
        print("❌ nvvidconv_out -> nvosd")
        return None
    
    if not nvosd.link(sink):
        print("❌ nvosd -> sink")
        return None
    
    print("\n✓ Пайплайн создан успешно!")
    print("=" * 60)
    
    return pipeline, appsrc


def check_gpu():
    """Проверка доступности GPU"""
    print("\n" + "=" * 60)
    print("Проверка системы")
    print("=" * 60)
    
    # Проверяем NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU обнаружена")
            # Парсим вывод для получения информации
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("⚠️ NVIDIA GPU не обнаружена")
    except FileNotFoundError:
        print("⚠️ nvidia-smi не найден")
    
    # Проверяем GStreamer
    try:
        result = subprocess.run(['gst-launch-1.0', '--version'], 
                              capture_output=True, 
                              text=True)
        print(f"✓ GStreamer: {result.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️ GStreamer не найден")
    
    # Проверяем папку с изображениями
    if os.path.exists(IMAGE_FOLDER):
        num_images = len(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
        num_images += len(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
        print(f"✓ Папка '{IMAGE_FOLDER}': {num_images} изображений")
    else:
        print(f"⚠️ Папка '{IMAGE_FOLDER}' не найдена")
    
    print("=" * 60)


def main():
    global feeder
    
    # Проверяем систему
    check_gpu()
    
    # Инициализация GStreamer
    print("\nИнициализация GStreamer...")
    Gst.init(None)
    
    # Инициализируем фидер
    try:
        feeder = RingBufferFeeder(IMAGE_FOLDER, WIDTH, HEIGHT)
    except Exception as e:
        print(f"❌ Ошибка фидера: {e}")
        return
    
    # Создаем пайплайн
    result = create_pipeline()
    if not result:
        print("❌ Не удалось создать пайплайн")
        return
    
    pipeline, appsrc = result
    
    # Подключаем callback
    appsrc.connect("need-data", need_data)
    
    # Настраиваем bus
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    # Главный цикл
    loop = GLib.MainLoop()
    bus.connect("message", bus_call, loop)
    
    # Запускаем
    print("\n" + "=" * 60)
    print("Запуск пайплайна...")
    print("=" * 60)
    
    pipeline.set_state(Gst.State.PLAYING)
    
    print("\nПапплайн запущен!")
    print("Управление: Ctrl+C для остановки")
    print("-" * 60)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nОстановка...")
    
    # Очистка
    pipeline.set_state(Gst.State.NULL)
    
    # Статистика
    elapsed = time.time() - start_time
    if elapsed > 0:
        print("\n" + "=" * 60)
        print("Статистика")
        print("=" * 60)
        print(f"Всего кадров: {frame_count}")
        print(f"Время: {elapsed:.2f} секунд")
        print(f"FPS: {frame_count/elapsed:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()