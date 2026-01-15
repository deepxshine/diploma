from ultralytics import YOLO
import os

dataset_path = os.path.abspath('dataset')

model = YOLO('yolov8-rail.yaml')
model.load('yolov8n_mono.pt')

if __name__ == '__main__':
	model.train(
		data=f'{dataset_path}/data.yaml',
		epochs=100,
		imgsz=(512, 256),
		batch=5,
		device='mps',
		project='rail_project',
		name='mono_mvp_v2',
		rect=True,
		nbs=64,
		plots=True,
		overlap_mask=False,
	)
