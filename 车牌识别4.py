import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from queue import Queue, Empty
import time
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSplitter,
                             QGroupBox, QDoubleSpinBox, QCheckBox, QToolBar,
                             QStatusBar, QFileDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGraphicsDropShadowEffect,
                             QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re


class SafeUIAccessor:
	"""UI对象安全访问器 - 防止访问已删除的对象"""

	def __init__(self, main_window):
		self.main_window = main_window
		self._cache = {}

	def get(self, name):
		"""安全获取UI组件"""
		try:
			if name in self._cache:
				return self._cache[name]

			if hasattr(self.main_window, name):
				obj = getattr(self.main_window, name)
				# 验证对象是否有效
				if hasattr(obj, 'isWidgetType') and obj.isWidgetType():
					self._cache[name] = obj
					return obj
			return None
		except:
			return None

	def is_valid(self, name):
		"""检查UI组件是否有效"""
		try:
			obj = self.get(name)
			if obj is None:
				return False
			# 尝试访问对象属性来验证它是否有效
			_ = obj.objectName()
			return True
		except:
			return False


class ModernButton(QPushButton):
	"""现代化按钮控件"""

	def __init__(self, text, icon_path=None):
		super().__init__(text)
		self.setMinimumHeight(40)
		self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))


class OCRProcessorThread(QThread):
	"""OCR识别处理线程"""
	ocr_result = pyqtSignal(dict)  # 发送OCR识别结果
	queue_size_updated = pyqtSignal(int)  # 发送队列大小

	def __init__(self, parent=None):
		super().__init__(parent)
		self.ocr = self._init_ocr()
		self.plate_queue = Queue()
		self.running = False
		self.processed_plates = set()  # 已处理的车牌集合

	def _init_ocr(self):
		"""初始化PaddleOCR - 完全兼容2.7.0.3版本"""
		try:
			print("正在初始化PaddleOCR...")
			# 使用最稳定的参数组合
			ocr = PaddleOCR(
				lang="ch",
				use_angle_cls=True,
				enable_mkldnn=False,  # 禁用MKLDNN
				use_gpu=False,  # 强制CPU
				cpu_threads=4,
				show_log=False,
				det=True,  # 启用检测（需要）
				rec=True  # 启用识别
			)
			print("✅ PaddleOCR初始化成功")
			return ocr
		except Exception as e:
			print(f"❌ PaddleOCR初始化失败: {e}")
			traceback.print_exc()
			return None

	def run(self):
		"""线程运行函数 - 修复空队列异常处理"""
		if self.ocr is None:
			print("OCR引擎未初始化，线程无法运行")
			return

		self.running = True
		print("OCR处理线程已启动")

		while self.running:
			try:
				plate_data = self.plate_queue.get(timeout=0.1)
				self.queue_size_updated.emit(self.plate_queue.qsize())

				if plate_data is None:
					continue

				plate_img, bbox, timestamp, frame_idx = plate_data
				plate_text, ocr_conf = self.recognize_plate(plate_img)

				# 修改：只进行去重检查，不验证车牌格式有效性
				if plate_text and plate_text not in self.processed_plates:
					self.processed_plates.add(plate_text)
					result = {
						'plate_text': plate_text,
						'ocr_conf': ocr_conf,
						'bbox': bbox,
						'timestamp': timestamp,
						'frame_idx': frame_idx
					}
					self.ocr_result.emit(result)
					print(f"💡 识别到车牌: {plate_text} (置信度: {ocr_conf:.2f})")

			except Empty:
				# ✅ 队列为空是正常现象，静默处理
				time.sleep(0.01)
				continue
			except Exception as e:
				# 只有真正的异常才打印错误信息
				print(f"⚠️ OCR处理异常: {e}")
				time.sleep(0.01)
				continue

	def add_plate(self, plate_img, bbox, timestamp, frame_idx):
		"""添加车牌到处理队列"""
		if not self.running or self.ocr is None:
			return
		if self.plate_queue.qsize() < 1000:
			# 修复：使用参数frame_idx而不是self.frame_idx
			self.plate_queue.put((plate_img, bbox, timestamp, frame_idx))
			self.queue_size_updated.emit(self.plate_queue.qsize())
			print(f"📥 添加车牌到队列，当前队列大小: {self.plate_queue.qsize()}")

	def stop(self):
		"""停止线程"""
		print("正在停止OCR处理线程...")
		self.running = False
		self.wait()
		print("OCR处理线程已停止")

	def recognize_plate(self, plate_img):
		"""识别车牌文字 - 添加格式验证和字母转换"""
		if self.ocr is None:
			print("❌ OCR引擎为None")
			return None, 0.0

		try:
			# 预处理
			h, w = plate_img.shape[:2]
			target_h, target_w = 60, 180

			if h > 0 and w > 0:
				scale = min(target_w / w, target_h / h)
				new_w, new_h = int(w * scale), int(h * scale)
				plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

				top = (target_h - new_h) // 2
				bottom = target_h - new_h - top
				left = (target_w - new_w) // 2
				right = target_w - new_w - left

				plate_img = cv2.copyMakeBorder(
					plate_img, top, bottom, left, right,
					cv2.BORDER_CONSTANT, value=(128, 128, 128)
				)

			# 图像增强
			plate_img = cv2.convertScaleAbs(plate_img, alpha=1.3, beta=15)
			plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

			# OCR识别
			result = self.ocr.ocr(plate_img_rgb, cls=True)

			# 打印调试信息
			print(f"🔍 OCR返回结果: {result}")

			# 严格检查返回格式（PaddleOCR 2.7.0.3）
			if result is None:
				print("❌ OCR返回None")
				return None, 0.0

			if not isinstance(result, list):
				print(f"❌ OCR返回类型错误: {type(result)}")
				return None, 0.0

			if len(result) == 0:
				print("❌ OCR返回空列表")
				return None, 0.0

			# result[0]可能是None（没有检测到文本）
			if result[0] is None:
				print("⚠️ OCR未检测到任何文本")
				return None, 0.0

			# result[0]是列表，包含多个检测框的结果
			detections = result[0]
			if not isinstance(detections, list) or len(detections) == 0:
				print(f"❌ 检测结果格式错误: {detections}")
				return None, 0.0

			# 遍历所有检测框，找到最可能的文本
			best_text = None
			best_conf = 0.0

			for detection in detections:
				if not isinstance(detection, list) or len(detection) < 2:
					continue

				# detection格式: [[[框坐标]], [文本, 置信度]]
				if len(detection) == 2:
					# 复杂格式: [框信息, [文本, 置信度]]
					text_info = detection[1]
					# 修复：支持list或tuple类型（PaddleOCR返回的是tuple）
					if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
						text = text_info[0]
						conf = float(text_info[1])
					else:
						continue
				else:
					continue

				# 清理识别结果 - 保留字母、数字和中文
				original_text = text
				text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text)

				print(f"📝 原始文本: {original_text} -> 清理后: {text} (置信度: {conf})")

				# 选择最优结果（置信度最高且符合基本长度要求）
				# 修改：只检查长度，不验证具体车牌格式
				if len(text) >= 6 and len(text) <= 8 and conf > best_conf:
					best_text = text
					best_conf = conf

			if best_text:
				print(f"✅ 选择最优结果: {best_text} (置信度: {best_conf})")
				return best_text, best_conf
			else:
				print("⚠️ 未找到长度在6-8之间的有效文本")
				return None, 0.0

		except Exception as e:
			print(f"❌ OCR识别失败: {e}")
			traceback.print_exc()

		return None, 0.0

	def is_valid_plate(self, text):
		"""验证车牌格式 - 此函数不再使用，保留仅供参考"""
		# 此函数已废弃，不再进行车牌格式验证
		# 所有通过OCR识别且长度合理的文本都会被接受
		return True

	def clear_processed_plates(self):
		"""清空已处理车牌集合"""
		self.processed_plates.clear()


class VideoProcessorThread(QThread):
	"""视频处理线程"""
	frame_received = pyqtSignal(np.ndarray, list)  # 原始帧 + 检测框信息
	fps_updated = pyqtSignal(float)

	def __init__(self, detector, ocr_thread, parent=None):
		super().__init__(parent)
		self.detector = detector
		self.ocr_thread = ocr_thread
		self.video_path = None
		self.running = False
		self.cap = None
		self.fps = 0
		self.frame_idx = 0
		self.conf_threshold = 0.5

		# 速率限制（每秒最多5张）
		self.last_check_time = time.time()
		self.plates_added_this_second = 0

	def set_video(self, video_path, conf_threshold=0.5):
		self.video_path = video_path
		self.conf_threshold = conf_threshold

	def run(self):
		if self.video_path is None:
			return

		self.cap = cv2.VideoCapture(self.video_path)
		if not self.cap.isOpened():
			print(f"❌ 无法打开视频源: {self.video_path}")
			return

		self.running = True
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		if self.fps == 0:
			self.fps = 30

		fps_counter = deque(maxlen=30)
		fps_timer = cv2.getTickCount()

		print(f"视频处理线程已启动，FPS: {self.fps:.1f}")

		while self.running:
			ret, frame = self.cap.read()
			if not ret:
				break

			self.frame_idx += 1

			# FPS计算
			current_time = cv2.getTickCount()
			time_diff = (current_time - fps_timer) / cv2.getTickFrequency()
			fps_timer = current_time
			if time_diff > 0:
				fps_counter.append(1 / time_diff)
				if len(fps_counter) > 0:
					self.fps_updated.emit(np.mean(fps_counter))

			# 检测车牌
			results = self.detector.detect_yolo(frame, self.conf_threshold)
			detections_for_draw = []

			# 重置速率限制
			current_time_sec = time.time()
			if current_time_sec - self.last_check_time >= 1.0:
				self.last_check_time = current_time_sec
				self.plates_added_this_second = 0

			# 处理每个检测框
			if results:
				for result in results:
					boxes = result.boxes
					if boxes is None:
						continue

					for box in boxes:
						x1, y1, x2, y2 = map(int, box.xyxy[0])
						conf = float(box.conf[0])

						if conf < self.conf_threshold:
							continue

						plate_img = frame[y1:y2, x1:x2]
						if plate_img.size == 0:
							continue

						# 速率限制
						can_add_to_ocr = False
						if self.plates_added_this_second < 5:
							timestamp = datetime.now().strftime("%H:%M:%S")
							if self.ocr_thread is not None and self.ocr_thread.running:
								self.ocr_thread.add_plate(plate_img.copy(), (x1, y1, x2, y2), timestamp, self.frame_idx)
								self.plates_added_this_second += 1
								can_add_to_ocr = True

						plate_text = '识别中...' if can_add_to_ocr else '等待识别'

						detections_for_draw.append({
							'bbox': (x1, y1, x2, y2),
							'conf': conf,
							'plate_text': plate_text,
							'timestamp': timestamp
						})

			# 发送帧和检测信息
			self.frame_received.emit(frame, detections_for_draw)

			# 控制帧率
			if self.fps > 0:
				time.sleep(1 / (self.fps * 2))

		self.cap.release()
		print("视频处理线程已停止")

	def stop(self):
		print("正在停止视频处理线程...")
		self.running = False
		self.wait()
		print("视频处理线程已停止")


class LicensePlateDetector:
	"""车牌检测器"""

	def __init__(self):
		self.yolo_model = None

	def load_model(self, model_path):
		"""加载YOLO模型"""
		try:
			self.yolo_model = YOLO(model_path)
			print(f"✅ YOLO模型加载成功: {model_path}")
			return True
		except Exception as e:
			print(f"❌ YOLO模型加载失败: {e}")
			traceback.print_exc()
			return False

	def detect_yolo(self, frame, conf_threshold=0.5):
		"""仅进行YOLO检测"""
		if self.yolo_model is None:
			return []
		return self.yolo_model(frame, verbose=False, conf=conf_threshold)


class LicensePlateApp(QMainWindow):
	"""主应用程序窗口"""

	def __init__(self):
		super().__init__()
		self.detector = LicensePlateDetector()
		self.video_thread = None
		self.ocr_thread = None
		self.current_frame = None
		self.detected_plates = set()
		self.ui_accessor = SafeUIAccessor(self)  # UI安全访问器

		print("正在初始化UI...")
		self.init_ui()
		self.apply_modern_style()
		print("✅ 主窗口初始化完成")

	def init_ui(self):
		"""初始化UI"""
		self.setWindowTitle("🚗 超级智能车牌识别系统")
		self.setMinimumSize(1600, 900)

		central_widget = QWidget()
		self.setCentralWidget(central_widget)

		main_layout = QHBoxLayout(central_widget)
		main_layout.setContentsMargins(0, 0, 0, 0)

		splitter = QSplitter(Qt.Orientation.Horizontal)

		# 创建面板
		left_panel = self.create_left_panel()
		right_panel = self.create_right_panel()

		splitter.addWidget(left_panel)
		splitter.addWidget(right_panel)
		splitter.setSizes([500, 1100])
		splitter.setHandleWidth(2)

		main_layout.addWidget(splitter)

		self.create_toolbar()
		self.create_status_bar()

	def create_left_panel(self):
		"""创建左侧面板"""
		left_widget = QWidget()
		left_layout = QVBoxLayout(left_widget)
		left_layout.setContentsMargins(15, 15, 15, 15)

		# 标题
		title_label = QLabel("📋 车牌识别结果")
		title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 15px;")
		left_layout.addWidget(title_label)

		# 统计信息
		stats_group = QGroupBox("统计信息")
		stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
		stats_layout = QHBoxLayout(stats_group)

		self.total_plates_label = QLabel("总车牌数: 0")
		self.total_plates_label.setFont(QFont("Segoe UI", 12))
		self.fps_label = QLabel("FPS: 0")
		self.fps_label.setFont(QFont("Segoe UI", 12))

		stats_layout.addWidget(self.total_plates_label)
		stats_layout.addWidget(self.fps_label)

		left_layout.addWidget(stats_group)

		# 车牌表格
		self.plate_table = QTableWidget()
		self.plate_table.setColumnCount(4)
		self.plate_table.setHorizontalHeaderLabels(["车牌号", "检测置信度", "OCR置信度", "时间"])
		self.plate_table.horizontalHeader().setStretchLastSection(True)
		self.plate_table.setAlternatingRowColors(True)
		self.plate_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
		self.plate_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

		self.plate_table.setStyleSheet("""
            QTableWidget {
                background-color: #2D2D2D;
                alternate-background-color: #363636;
                gridline-color: #444;
                border: none;
                border-radius: 8px;
            }
            QHeaderView::section {
                background-color: #3D3D3D;
                color: #E0E0E0;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)

		left_layout.addWidget(self.plate_table)

		# 控制按钮
		button_layout = QHBoxLayout()

		self.start_btn = ModernButton("▶ 开始识别")
		self.start_btn.clicked.connect(self.start_recognition_safe)
		self.start_btn.setEnabled(False)

		self.stop_btn = ModernButton("⏹ 停止识别")
		self.stop_btn.clicked.connect(self.stop_recognition_safe)
		self.stop_btn.setEnabled(False)

		self.clear_btn = ModernButton("🗑 清空记录")
		self.clear_btn.clicked.connect(self.clear_records)

		button_layout.addWidget(self.start_btn)
		button_layout.addWidget(self.stop_btn)
		button_layout.addWidget(self.clear_btn)

		left_layout.addLayout(button_layout)

		return left_widget

	def create_right_panel(self):
		"""创建右侧面板"""
		right_widget = QWidget()
		right_layout = QVBoxLayout(right_widget)
		right_layout.setContentsMargins(15, 15, 15, 15)

		# 标题
		video_title = QLabel("🎬 实时视频画面")
		video_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		video_title.setStyleSheet("color: #E0E0E0; margin-bottom: 15px;")
		right_layout.addWidget(video_title)

		# 视频显示区域
		self.video_label = QLabel()
		self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border: 2px solid #444;
                border-radius: 12px;
                qproperty-alignment: AlignCenter;
            }
        """)
		self.video_label.setMinimumSize(800, 600)
		self.video_label.setText("📹 请先加载视频文件或摄像头")
		right_layout.addWidget(self.video_label)

		# 参数设置
		params_group = QGroupBox("检测参数设置")
		params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
		params_layout = QHBoxLayout(params_group)

		# 置信度阈值
		conf_layout = QVBoxLayout()
		conf_label = QLabel("YOLO置信度阈值:")
		self.conf_spin = QDoubleSpinBox()
		self.conf_spin.setRange(0.1, 1.0)
		self.conf_spin.setValue(0.5)
		self.conf_spin.setSingleStep(0.05)
		conf_layout.addWidget(conf_label)
		conf_layout.addWidget(self.conf_spin)

		# 显示选项
		show_layout = QVBoxLayout()
		self.show_bbox_check = QCheckBox("显示检测框")
		self.show_bbox_check.setChecked(True)
		self.show_conf_check = QCheckBox("显示置信度")
		self.show_conf_check.setChecked(True)
		show_layout.addWidget(self.show_bbox_check)
		show_layout.addWidget(self.show_conf_check)

		params_layout.addLayout(conf_layout)
		params_layout.addLayout(show_layout)

		return right_widget

	def create_toolbar(self):
		"""创建工具栏"""
		toolbar = QToolBar()
		toolbar.setIconSize(QSize(24, 24))
		toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2D2D2D;
                border: none;
                padding: 8px;
                spacing: 10px;
            }
            QToolBar::separator {
                background-color: #444;
                width: 2px;
                margin: 5px;
            }
        """)

		load_model_action = ModernButton("📂 加载YOLO模型")
		load_model_action.clicked.connect(self.load_model)
		toolbar.addWidget(load_model_action)

		toolbar.addSeparator()

		load_video_action = ModernButton("🎬 加载视频")
		load_video_action.clicked.connect(self.load_video)
		toolbar.addWidget(load_video_action)

		toolbar.addSeparator()

		load_camera_action = ModernButton("📹 打开摄像头")
		load_camera_action.clicked.connect(self.load_camera)
		toolbar.addWidget(load_camera_action)

		self.addToolBar(toolbar)

	def create_status_bar(self):
		"""创建状态栏"""
		self.status_bar = QStatusBar()
		self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border-top: 1px solid #444;
            }
        """)

		self.status_label = QLabel("就绪")
		self.status_bar.addWidget(self.status_label)

		self.model_label = QLabel("模型: 未加载")
		self.status_bar.addPermanentWidget(self.model_label)

		self.video_label_status = QLabel("视频: 未加载")
		self.status_bar.addPermanentWidget(self.video_label_status)

		self.pending_frames_label = QLabel("待处理: 0帧")
		self.status_bar.addPermanentWidget(self.pending_frames_label)

		self.setStatusBar(self.status_bar)

	def apply_modern_style(self):
		"""应用现代化样式"""
		style = """
            QMainWindow {
                background-color: #252525;
            }
            QWidget {
                background-color: #252525;
                color: #E0E0E0;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4D4D4D;
            }
            QPushButton:pressed {
                background-color: #2D2D2D;
            }
            QPushButton:disabled {
                background-color: #1D1D1D;
                color: #666;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px;
            }
            QCheckBox {
                color: #E0E0E0;
            }
            QTableWidget::item:selected {
                background-color: #4D4D2D;
            }
            QScrollBar:vertical {
                background-color: #2D2D2D;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #5D5D5D;
                border-radius: 6px;
                min-height: 20px;
            }
        """
		self.setStyleSheet(style)

	def load_model(self):
		"""加载模型"""
		file_path, _ = QFileDialog.getOpenFileName(
			self, "选择YOLO模型", "", "模型文件 (*.pt *.onnx)")

		if file_path:
			self.status_label.setText("正在加载模型...")
			QApplication.processEvents()

			if self.detector.load_model(file_path):
				self.model_label.setText(f"模型: {Path(file_path).name}")
				self.status_label.setText("✅ YOLO模型加载成功")

				# 创建OCR线程（只创建一次）
				if self.ocr_thread is None:
					self.ocr_thread = OCRProcessorThread(self)  # 设置父对象
					if self.ocr_thread.ocr is not None:
						# 安全连接信号
						try:
							self.ocr_thread.ocr_result.disconnect()
						except:
							pass
						try:
							self.ocr_thread.queue_size_updated.disconnect()
						except:
							pass

						self.ocr_thread.ocr_result.connect(self.handle_ocr_result_safe)
						self.ocr_thread.queue_size_updated.connect(self.update_pending_frames)
						self.ocr_thread.start()
						self.status_label.setText("✅ PaddleOCR中文识别引擎已启动")
					else:
						QMessageBox.critical(self, "错误",
						                     "PaddleOCR初始化失败，请检查安装:\n\n"
						                     "pip install paddlepaddle paddleocr")
						return

				if hasattr(self, 'video_path'):
					self.start_btn.setEnabled(True)
			else:
				QMessageBox.critical(self, "错误", "❌ YOLO模型加载失败")
				self.status_label.setText("模型加载失败")

	def load_video(self):
		"""加载视频"""
		file_path, _ = QFileDialog.getOpenFileName(
			self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")

		if file_path:
			self.video_path = file_path
			self.video_label_status.setText(f"视频: {Path(file_path).name}")
			self.status_label.setText("✅ 视频加载成功")

			if self.detector.yolo_model is not None:
				self.start_btn.setEnabled(True)

	def load_camera(self):
		"""加载摄像头"""
		camera_id, ok = QInputDialog.getInt(self, "选择摄像头", "摄像头ID (0为默认):", 0, 0, 10)
		if ok:
			self.video_path = camera_id
			self.video_label_status.setText(f"视频: 摄像头 {camera_id}")
			self.status_label.setText("✅ 摄像头已连接")

			if self.detector.yolo_model is not None:
				self.start_btn.setEnabled(True)

	def start_recognition_safe(self):
		"""线程安全的启动识别"""
		try:
			self.start_recognition()
		except RuntimeError as e:
			print(f"RuntimeError in start_recognition: {e}")
			QMessageBox.critical(self, "错误", "UI组件已失效，请重启程序")
		except Exception as e:
			print(f"异常 in start_recognition: {e}")

	def start_recognition(self):
		"""开始识别"""
		if not hasattr(self, 'video_path'):
			QMessageBox.warning(self, "警告", "请先加载视频或摄像头")
			return

		if self.ocr_thread is None or self.ocr_thread.ocr is None:
			QMessageBox.warning(self, "警告", "OCR引擎未就绪，请重新加载模型")
			return

		# 停止之前的线程
		if self.video_thread and self.video_thread.isRunning():
			self.video_thread.stop()

		# 清空已处理集合
		self.detected_plates.clear()
		if self.ocr_thread:
			self.ocr_thread.clear_processed_plates()

		# 获取配置
		try:
			conf_threshold = self.conf_spin.value()
		except RuntimeError:
			conf_threshold = 0.5

		# 创建新视频处理线程
		self.video_thread = VideoProcessorThread(self.detector, self.ocr_thread, self)
		self.video_thread.set_video(self.video_path, conf_threshold)

		# 安全连接信号
		try:
			self.video_thread.frame_received.disconnect()
		except:
			pass
		try:
			self.video_thread.fps_updated.disconnect()
		except:
			pass

		self.video_thread.frame_received.connect(self.update_frame_safe)
		self.video_thread.fps_updated.connect(self.update_fps_safe)

		# 更新UI状态
		try:
			self.start_btn.setEnabled(False)
			self.stop_btn.setEnabled(True)
			self.status_label.setText("🔍 正在识别中...")
		except RuntimeError:
			pass

		# 启动线程
		self.video_thread.start()
		print("✅ 识别已启动")

	def stop_recognition_safe(self):
		"""线程安全的停止识别"""
		try:
			self.stop_recognition()
		except RuntimeError as e:
			print(f"RuntimeError in stop_recognition: {e}")
		except Exception as e:
			print(f"异常 in stop_recognition: {e}")

	def stop_recognition(self):
		"""停止识别"""
		if self.video_thread:
			self.video_thread.stop()
			self.video_thread = None

		try:
			self.start_btn.setEnabled(True)
			self.stop_btn.setEnabled(False)
			self.status_label.setText("⏹️ 识别已停止")
		except RuntimeError:
			pass

	def update_frame_safe(self, frame, detections):
		"""线程安全的更新帧"""
		try:
			self.update_frame(frame, detections)
		except RuntimeError as e:
			print(f"RuntimeError in update_frame: {e}")
		except Exception as e:
			print(f"异常 in update_frame: {e}")

	def update_frame(self, frame, detections):
		"""更新视频帧"""
		if frame is None:
			return

		self.current_frame = frame.copy()

		# 绘制检测结果
		# ✅ 安全访问checkbox
		show_bbox = self.ui_accessor.is_valid('show_bbox_check') and self.show_bbox_check.isChecked()
		show_conf = self.ui_accessor.is_valid('show_conf_check') and self.show_conf_check.isChecked()

		if show_bbox:
			for det in detections:
				x1, y1, x2, y2 = det['bbox']
				plate_text = det['plate_text']
				conf = det['conf']

				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

				label = f"{plate_text}"
				if show_conf:
					label += f" ({conf:.2f})"

				(text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
				cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
				cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		# 转换为Qt图像
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_frame.shape
		bytes_per_line = ch * w

		qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
		pixmap = QPixmap.fromImage(qt_image)

		# ✅ 安全访问video_label
		if self.ui_accessor.is_valid('video_label'):
			# 缩放以适应标签
			scaled_pixmap = pixmap.scaled(
				self.video_label.size(),
				Qt.AspectRatioMode.KeepAspectRatio,
				Qt.TransformationMode.SmoothTransformation
			)
			self.video_label.setPixmap(scaled_pixmap)

	def handle_ocr_result_safe(self, result):
		"""线程安全的处理OCR结果"""
		try:
			self.handle_ocr_result(result)
		except RuntimeError as e:
			print(f"RuntimeError in handle_ocr_result: {e}")
		except Exception as e:
			print(f"异常 in handle_ocr_result: {e}")

	def handle_ocr_result(self, result):
		"""处理OCR识别结果 - 包含格式验证和字母转换"""
		# 提取车牌文本和置信度
		raw_text = result['plate_text']
		conf = result['ocr_conf']

		# 第一步：清理特殊字符（删除"·"和空格）
		cleaned_text = raw_text.replace('·', '').replace(' ', '')
		print(f"📝 原始文本: '{raw_text}' -> 清理后: '{cleaned_text}'")

		# 第二步：验证格式（一个中文 + 5或6个字母数字）
		if not self.validate_plate_format(cleaned_text):
			print(f"❌ 车牌格式验证失败: '{cleaned_text}'")
			return

		# 第三步：字母转换（l→1, o→0, L→1, O→0）
		converted_text = self.convert_letters(cleaned_text)
		print(f"🔤 字母转换: '{cleaned_text}' -> '{converted_text}'")

		# 第四步：检查UI去重
		if converted_text in self.detected_plates:
			print(f"📌 UI中已存在车牌，跳过更新: '{converted_text}'")
			return

		# 验证通过，添加到UI
		self.detected_plates.add(converted_text)
		print(f"✅ 添加新车牌到UI: '{converted_text}'")

		# 添加到表格（更新result中的文本为转换后的）
		result['plate_text'] = converted_text
		row = self.plate_table.rowCount()
		self.plate_table.insertRow(row)

		# 车牌号
		plate_item = QTableWidgetItem(converted_text)
		plate_item.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
		plate_item.setForeground(QColor("#4FC3F7"))
		self.plate_table.setItem(row, 0, plate_item)

		# 检测置信度
		conf_item = QTableWidgetItem(f"{0.95:.2f}")
		conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
		self.plate_table.setItem(row, 1, conf_item)

		# OCR置信度
		ocr_conf_item = QTableWidgetItem(f"{conf:.2f}")
		ocr_conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
		self.plate_table.setItem(row, 2, ocr_conf_item)

		# 时间
		time_item = QTableWidgetItem(result['timestamp'])
		time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
		self.plate_table.setItem(row, 3, time_item)

		# 自动滚动到最新行
		self.plate_table.scrollToBottom()

		# 更新统计
		self.total_plates_label.setText(f"📊 总车牌数: {self.plate_table.rowCount()}")

	def validate_plate_format(self, text):
		"""验证车牌格式：一个中文 + 5或6个字母数字"""
		if not text or len(text) < 6 or len(text) > 7:
			return False

		# 检查第一个字符是中文（省份简称）
		provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
		if text[0] not in provinces:
			return False

		# 检查后面5-6个字符是字母或数字
		if not re.match(r'^[' + provinces + r'][A-Z0-9]{5,6}$', text):
			return False

		return True

	def convert_letters(self, text):
		"""转换字母：l→1, L→1, o→0, O→0"""
		# 只转换省份后的字符（不转换中文）
		if len(text) < 2:
			return text

		# 保留省份字符，转换后面的字符
		province = text[0]
		plate_number = text[1:]

		# 替换字母
		plate_number = plate_number.replace('l', '1').replace('L', '1')
		plate_number = plate_number.replace('o', '0').replace('O', '0')

		return province + plate_number

	def update_fps_safe(self, fps):
		"""线程安全的更新FPS"""
		try:
			if self.ui_accessor.is_valid('fps_label'):
				self.update_fps(fps)
		except RuntimeError:
			pass

	def update_fps(self, fps):
		"""更新FPS"""
		self.fps_label.setText(f"🎬 FPS: {fps:.1f}")

	def update_pending_frames(self, queue_size):
		"""更新待处理帧数显示"""
		self.pending_frames_label.setText(f"⏳ 待处理: {queue_size}帧")

	def clear_records(self):
		"""清空记录"""
		self.plate_table.setRowCount(0)
		self.total_plates_label.setText("📊 总车牌数: 0")
		self.detected_plates.clear()
		if self.ocr_thread:
			self.ocr_thread.clear_processed_plates()

	def closeEvent(self, event):
		"""关闭事件"""
		print("正在关闭应用程序...")

		# 停止线程
		if self.video_thread and self.video_thread.isRunning():
			self.video_thread.stop()
		if self.ocr_thread and self.ocr_thread.isRunning():
			self.ocr_thread.stop()

		print("✅ 应用程序已关闭")
		event.accept()


def main():
	print("=" * 60)
	print("应用程序启动...")
	print("=" * 60)

	app = QApplication(sys.argv)

	# 设置应用字体
	font = QFont("Microsoft YaHei", 10)
	app.setFont(font)

	# 创建窗口
	window = LicensePlateApp()
	window.show()
	print("✅ 主窗口已显示")

	sys.exit(app.exec())


if __name__ == "__main__":
	main()