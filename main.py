import sys
import time
import threading
import numpy as np
import cv2
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# --- КОНФИГУРАЦИЯ ---
WIDTH = 1920
HEIGHT = 1080
FPS = 30  # Для прототипа ставим 30, чтобы успевал GUI. В headless можно гнать быстрее.
# Список файлов-имитаторов (фото с iPhone)
IMAGE_FILES = ["sample_1.jpg", "sample_2.jpg"]


class RingBufferFeeder:
	"""
	Эмуляция драйвера камеры и кольцевого буфера.
	Хранит декодированные изображения в RAM и отдает их по запросу.
	"""

	def __init__(self, file_paths, target_w, target_h):
		self.frames_buffer = []
		self.current_idx = 0
		self.lock = threading.Lock()

		print(f"[RAM] Загрузка {len(file_paths)} кадров в ОЗУ...")
		for fp in file_paths:
			img = cv2.imread(fp)
			if img is None:
				print(f"Ошибка чтения {fp}")
				continue

			# Ресайз сразу при загрузке, чтобы не тратить CPU в цикле (эмуляция работы сенсора)
			img = cv2.resize(img, (target_w, target_h))

			# GStreamer любит RGBA, OpenCV дает BGR. Конвертируем заранее.
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

			# Превращаем в байты для отправки
			self.frames_buffer.append(img.tobytes())

		if not self.frames_buffer:
			raise Exception("Не удалось загрузить ни одного изображения!")
		print(
			f"[RAM] Данные загружены. Размер буфера: {len(self.frames_buffer)} кадров."
			)

	def get_next_frame_bytes(self):
		"""Возвращает следующий кадр из кольцевого буфера"""
		with self.lock:
			data = self.frames_buffer[self.current_idx]
			# Двигаем индекс по кругу
			self.current_idx = (self.current_idx + 1) % len(self.frames_buffer)
		return data


# Глобальный фидер
feeder = None


def need_data(src, length):
	"""
	Callback, который вызывает appsrc, когда ему нужны данные.
	"""
	global feeder
	try:
		# 1. Берем данные из нашего "ОЗУ"
		data = feeder.get_next_frame_bytes()

		# 2. Оборачиваем в GstBuffer
		buf = Gst.Buffer.new_allocate(None, len(data), None)
		buf.fill(0, data)

		# 3. Важно: выставляем таймстемпы (иначе muxer может зависнуть)
		# В простом варианте appsrc может сам считать время, если do-timestamp=true

		# 4. Пушим буфер в пайплайн
		retval = src.emit('push-buffer', buf)

		if retval != Gst.FlowReturn.OK:
			print(f"Error pushing buffer: {retval}")

	except Exception as e:
		print(f"Error in need_data: {e}")


def main():
	global feeder
	Gst.init(None)

	# Инициализируем наш "буфер"
	# Создай пару картинок sample_1.jpg или поменяй пути
	try:
		feeder = RingBufferFeeder(IMAGE_FILES, WIDTH, HEIGHT)
	except Exception as e:
		print(e)
		return

	# --- СОЗДАНИЕ ПАЙПЛАЙНА ---
	print("[Pipeline] Создание элементов...")
	pipeline = Gst.Pipeline()

	# 1. AppSrc - входная точка из Python-скрипта
	appsrc = Gst.ElementFactory.make("appsrc", "ram-source")

	# Настройка форматов данных (Caps). Это критически важно для appsrc.
	# Мы говорим пайплайну: "Жди сырое видео, RGBA, 1920x1080"
	caps = Gst.Caps.from_string(
		f"video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate={FPS}/1"
	)
	appsrc.set_property("caps", caps)
	appsrc.set_property("format", Gst.Format.TIME)
	# do-timestamp=True заставляет appsrc самому расставлять временные метки
	# это упрощает код, но для жесткого Real-Time (п. 3.3) лучше считать самим.
	appsrc.set_property("do-timestamp", True)

	# Подключаем сигналы
	appsrc.connect("need-data", need_data)

	# 2. Videoconvert (CPU) -> NVVideoConvert (Загрузка в память GPU)
	# Сначала конвертер, чтобы убедиться, что формат понятен, затем загрузка в NVMM
	nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "uploader")

	# 3. Muxer (Агрегация в батчи, как в п. 3.2.4)
	streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
	streammux.set_property('width', WIDTH)
	streammux.set_property('height', HEIGHT)
	streammux.set_property(
		'batch-size', 1
		)  # Пока без нейронки ставим 1, или 16 если хочешь проверить нагрузку памяти
	streammux.set_property('batched-push-timeout', 4000000)

	# 4. Обработка (Здесь мы УБРАЛИ nvinfer)
	# Можно поставить nvvideoconvert для преобразования цветового пространства перед выводом
	nvvidconv_out = Gst.ElementFactory.make("nvvideoconvert", "converter-out")

	# 5. OSD (Отрисовка текста/рамок, если бы они были)
	nvosd = Gst.ElementFactory.make("nvdsosd", "osd")

	# 6. Вывод (Sink)
	# nveglglessink показывает окно. Для работы без монитора используй fakesink
	sink = Gst.ElementFactory.make("nveglglessink", "video-renderer")
	sink.set_property(
		"sync", False
		)  # False = пытаться играть максимально быстро (тест пропускной способности)

	# Добавляем в пайплайн
	pipeline.add(appsrc)
	pipeline.add(nvvidconv_src)
	pipeline.add(streammux)
	pipeline.add(nvvidconv_out)
	pipeline.add(nvosd)
	pipeline.add(sink)

	# Линковка
	# appsrc -> nvvidconv_src
	appsrc.link(nvvidconv_src)

	# nvvidconv_src -> streammux
	sinkpad = streammux.get_request_pad("sink_0")
	srcpad = nvvidconv_src.get_static_pad("src")
	srcpad.link(sinkpad)

	# streammux -> nvvidconv_out -> nvosd -> sink
	streammux.link(nvvidconv_out)
	nvvidconv_out.link(nvosd)
	nvosd.link(sink)

	# --- ЗАПУСК ---
	loop = GLib.MainLoop()
	pipeline.set_state(Gst.State.PLAYING)

	print("[Pipeline] Запущен. Окно должно открыться.")
	try:
		loop.run()
	except KeyboardInterrupt:
		pass

	pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
	main()