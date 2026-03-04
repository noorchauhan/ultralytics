


'''
cls_feat [B,C,H,W]
  → calculate_anomaly_score()          # 始终执行：cosine sim → [H*W] ∈ [0,1]
  → pf_score_spatial > conf → mask     # 始终执行：过滤候选区域

  ┌─ anomaly_mode=True  ─────────────────────────────────────────────┐
  │  logit(anomaly_score[mask]) → [B, 1, N]                          │
  │  _inference: sigmoid → 还原回 anomaly score 作为置信度            │
  │  postprocess: split([4, nc=1])                                   │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ anomaly_mode=False ─────────────────────────────────────────────┐
  │  vocab_linear(cls_feat[:, mask]) → [B, N, original_nc]           │
  │  _inference: sigmoid → 原始类别置信度                             │
  │  postprocess: split([4, nc=original_nc])                         │
  └──────────────────────────────────────────────────────────────────┘


'''

import os
from ultralytics.models.yolo.model import YOLOAnomaly
from ultralytics.engine import model

def test_leather_ad_func():

	category="leather"
	# category="grid"
	# category="bottle"
	category="cable"
	# category="carpet"
	# category="toothbrush"
	# category="metal_nut"
	# category="pill"
	train_dir=f"/Users/louis/workspace/public_datasets/mvtec_anomaly_detection/{category}/train/good"		
	test_dir=f"/Users/louis/workspace/public_datasets/mvtec_anomaly_detection/{category}/test"
	imgsz=640

	# ── Build model ──────────────────────────────────────────────────────────
	# model = YOLOAnomaly("yolo26x-seg.pt")
	# model = YOLOAnomaly("yoloe-26x-seg.pt")
	model = YOLOAnomaly("yolov8x.pt")
	# model = YOLOAnomaly("yolo11n-seg.pt")
	model.setup(["anomaly"], conf=0.0001)

	# ── Populate memory bank from support set (normal images) ─────────────
	support_images = [
		os.path.join(train_dir, img)
		for img in os.listdir(train_dir)
		if img.endswith((".png", ".jpg", ".jpeg"))
	][:10]
	model.load_support_set(support_images, imgsz=imgsz)

	model.set_mode("anomaly")
	# ── Pick a random test image ─────────────────────────────────────────
	def get_random_one_from_dir(dir_path):
		import random
		all_files=[]
		for root, dirs, files in os.walk(dir_path):
			for file in files:
				if file.endswith((".png", ".jpg", ".jpeg")):
					all_files.append(os.path.join(root, file))
		return random.choice(all_files)

	test_img=get_random_one_from_dir(test_dir)


	def get_test_img_mask(test_img):
		import cv2
		import numpy as np
		mask_path=test_img.replace("test","ground_truth").replace(".png","_mask.png")

		if os.path.exists(mask_path):
			return mask_path
		else:
			return None 


	# ── Run anomaly detection ─────────────────────────────────────────────
	res=model.predict(test_img, conf=0.00000001, imgsz=imgsz)[0]

	res.save("./runs/temp/res.png")
	from ultra_ext.utils import keep_best_detection_per_class
	res1=keep_best_detection_per_class(res)
	res1.save("./runs/temp/res1.png")
	# concat images: vertical or grid, with optional title strip above each image
	def vertical_concat_images(img_paths, save_path="./runs/temp/concat.png", layout="grid", grid_cols=2, texts=None):
		from PIL import Image, ImageDraw, ImageFont

		# Keep input order. If a path is None or invalid, use a gray placeholder image.
		valid_images = [Image.open(p).convert("RGB") for p in img_paths if p and os.path.exists(p)]
		if valid_images:
			width = max(img.width for img in valid_images)
			default_height = max(1, int(sum(img.height for img in valid_images) / len(valid_images)))
		else:
			width, default_height = 640, 480

		resized_images = []
		for p in img_paths:
			if p and os.path.exists(p):
				img = Image.open(p).convert("RGB")
				resized_images.append(img.resize((width, int(img.height * width / img.width))))
			else:
				resized_images.append(Image.new("RGB", (width, default_height), (0, 0, 0)))

		# Build title strips (gray bars) above each image
		title_h = int(width*0.075)
		if texts is None:
			texts = []

		# Try to load a system font scaled to title_h, fallback to default
		font_size = max(10, int(title_h * 0.65))
		font = None
		try:
			# macOS font paths
			for font_path in ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", 
							   "/Library/Fonts/Helvetica.ttf", "/System/Library/Fonts/Arial.ttf"]:
				if os.path.exists(font_path):
					font = ImageFont.truetype(font_path, font_size)
					break
		except:
			pass
		if font is None:
			font = ImageFont.load_default()

		tiles = []
		for i, img in enumerate(resized_images):
			title = str(texts[i]) if i < len(texts) and texts[i] is not None else ""
			title_bar = Image.new("RGB", (width, title_h), (160, 160, 160))
			draw = ImageDraw.Draw(title_bar)
			# Vertically center text in title bar
			bbox = draw.textbbox((0, 0), title, font=font)
			text_h = bbox[3] - bbox[1]
			y = max(0, (title_h - text_h) // 2)
			draw.text((10, y), title, fill=(20, 20, 20), font=font)

			tile = Image.new("RGB", (width, title_h + img.height), (0, 0, 0))
			tile.paste(title_bar, (0, 0))
			tile.paste(img, (0, title_h))
			tiles.append(tile)

		if layout == "grid":
			grid_cols = max(1, int(grid_cols))
			n = len(tiles)
			rows = (n + grid_cols - 1) // grid_cols

			# Use unified cell height for a cleaner grid
			cell_h = max(img.height for img in tiles)
			new_img = Image.new("RGB", (width * grid_cols, cell_h * rows), (0, 0, 0))

			for i, img in enumerate(tiles):
				r, c = divmod(i, grid_cols)
				# Top align each tile in its cell
				new_img.paste(img, (c * width, r * cell_h))
		else:
			# Default vertical concat behavior
			total_height = sum(img.height for img in tiles)
			new_img = Image.new("RGB", (width, total_height))

			y_offset = 0
			for img in tiles:
				new_img.paste(img, (0, y_offset))
				y_offset += img.height

		new_img.save(save_path)

	
	test_img_mask=get_test_img_mask(test_img)


	vertical_concat_images([test_img,test_img_mask,"./runs/temp/res.png","./runs/temp/res1.png"],
						texts=["Test Image","Ground Truth Mask","AD Result (All Detections)","AD Result (Best per Class)"],
						  save_path="./runs/temp/concat.png", layout="grid", grid_cols=2)

	# from ultra_ext.utils import open_in_vscode
	# open_in_vscode("./runs/temp/concat.png")

test_leather_ad_func()
