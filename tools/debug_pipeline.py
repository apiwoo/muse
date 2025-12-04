# Project MUSE - Pipeline Debugger (High Precision)
# (C) 2025 MUSE Corp.
# Usage: python tools/debug_pipeline.py

import sys
import os
import time
import cv2
import numpy as np
from collections import deque

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, QMutex, QMutexLocker

import moderngl

# ==============================================================================
# [1] Profiler Class (Rolling Average Calculation)
# ==============================================================================
class Profiler:
    def __init__(self, name, window_size=60):
        self.name = name
        self.history = deque(maxlen=window_size)
        self.last_tick = time.perf_counter()

    def tick(self):
        """Mark start time"""
        self.last_tick = time.perf_counter()

    def tock(self):
        """Mark end time and record duration (ms)"""
        dt = (time.perf_counter() - self.last_tick) * 1000.0
        self.history.append(dt)
        return dt

    def get_avg(self):
        if not self.history: return 0.0
        return sum(self.history) / len(self.history)

    def get_fps(self):
        avg_ms = self.get_avg()
        if avg_ms == 0: return 0.0
        return 1000.0 / avg_ms

# ==============================================================================
# [2] Worker Thread (Camera Source)
# ==============================================================================
class CameraDebugWorker(QThread):
    def __init__(self, cam_id=0):
        super().__init__()
        self.cam_id = cam_id
        self.running = True
        self.mutex = QMutex()
        
        # Shared Data
        self.latest_frame = None
        self.frame_id = 0
        self.timestamp = 0.0

        # Profilers
        self.prof_io = Profiler("Camera I/O (Read)")
        self.prof_interval = Profiler("Frame Interval")

    def run(self):
        print(f"📷 [Worker] Opening Camera {self.cam_id}...")
        cap = cv2.VideoCapture(self.cam_id)
        
        # [CRITICAL] USB Bandwidth Fix (Force MJPG)
        # YUY2로 1080p 전송 시 USB 2.0 대역폭 초과로 5fps 제한 걸림 -> MJPG로 압축 전송
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   -> Setup Result: {int(actual_w)}x{int(actual_h)} @ {actual_fps}FPS")

        last_loop_time = time.perf_counter()

        while self.running and cap.isOpened():
            # 1. Measure Interval (Jitter)
            now = time.perf_counter()
            dt_interval = (now - last_loop_time) * 1000.0
            self.prof_interval.history.append(dt_interval)
            last_loop_time = now

            # 2. Measure I/O (Blocking Read)
            self.prof_io.tick()
            ret, frame = cap.read()
            io_time = self.prof_io.tock()

            if not ret:
                print("   ⚠️ Frame Drop (Empty)")
                continue

            # 3. Update Shared Memory
            with QMutexLocker(self.mutex):
                self.latest_frame = frame
                self.frame_id += 1
                self.timestamp = now

        cap.release()
        print("📷 [Worker] Camera Released.")

    def stop(self):
        self.running = False
        self.wait()

# ==============================================================================
# [3] OpenGL Widget (ModernGL Render)
# ==============================================================================
class DebugGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ctx = None
        self.texture = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.prof_upload = Profiler("Texture Upload")
        self.prof_draw = Profiler("GPU Draw")
        self.current_frame = None

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        print(f"🎨 [GL] Context: {self.ctx.version_code}")
        
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    vec4 c = texture(tex, v_texcoord);
                    f_color = vec4(c.b, c.g, c.r, 1.0); // BGR to RGB
                }
            """
        )
        
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_vert', 'in_texcoord')])

    def set_frame(self, frame):
        self.current_frame = frame
        self.update() # Trigger paintGL

    def paintGL(self):
        if self.ctx is None or self.current_frame is None: return

        # [FIX] OpenGL Context Binding Fix
        # 검은 화면 원인: Qt의 내부 Framebuffer를 잡지 못해서 엉뚱한 곳에 그림
        try:
            fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
            fbo.use()
        except:
            return

        # Measure 1: Texture Upload (CPU -> GPU)
        self.prof_upload.tick()
        h, w = self.current_frame.shape[:2]
        
        if self.texture is None or self.texture.width != w or self.texture.height != h:
            if self.texture: self.texture.release()
            self.texture = self.ctx.texture((w, h), 3, dtype='f1')
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Ensure contiguous (Important for speed)
        if not self.current_frame.flags['C_CONTIGUOUS']:
            self.current_frame = np.ascontiguousarray(self.current_frame)
            
        self.texture.write(self.current_frame)
        self.prof_upload.tock()

        # Measure 2: Draw Call
        self.prof_draw.tick()
        self.ctx.viewport = (0, 0, self.width(), self.height())
        self.ctx.clear(0.0, 0.0, 0.0)
        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.prof_draw.tock()

# ==============================================================================
# [4] Main Window (Coordinator)
# ==============================================================================
class DebugWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE Pipeline X-Ray")
        self.resize(1280, 720)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0,0,0,0)
        
        # Stats Label
        self.lbl_stats = QLabel("Initializing...")
        self.lbl_stats.setStyleSheet("background: #000; color: #0F0; font-family: Consolas; font-size: 14px; padding: 10px;")
        self.lbl_stats.setFixedHeight(120)
        layout.addWidget(self.lbl_stats)

        # OpenGL Widget
        self.gl_widget = DebugGLWidget()
        layout.addWidget(self.gl_widget)

        # Worker
        self.worker = CameraDebugWorker(cam_id=0) # 0번 카메라 (필요시 수정)
        self.worker.start()

        # UI Timer (Target 60 FPS update)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pipeline)
        self.timer.start(1) # 1ms (As fast as possible)

        self.last_frame_id = -1
        self.ui_fps_prof = Profiler("UI Refresh")
        
        # [New] Console Log Timer
        self.last_console_log_time = 0

    def update_pipeline(self):
        self.ui_fps_prof.tick()

        # 1. Fetch from Worker
        frame = None
        fid = -1
        t_created = 0
        
        # Lock은 정말 찰나의 순간만 잡습니다
        with QMutexLocker(self.worker.mutex):
            if self.worker.latest_frame is not None:
                # Reference copy (Zero cost)
                frame = self.worker.latest_frame
                fid = self.worker.frame_id
                t_created = self.worker.timestamp

        # 2. Render only if new
        if frame is not None and fid > self.last_frame_id:
            # Calculate Latency (Thread -> UI)
            latency_ms = (time.perf_counter() - t_created) * 1000.0
            
            # Draw stats on frame (CPU cost, but informative)
            self._draw_overlay(frame, latency_ms)
            
            # Send to GPU
            self.gl_widget.set_frame(frame)
            self.last_frame_id = fid

        # 3. Update Text Stats (Console + UI)
        self._update_stats_label()
        self.ui_fps_prof.tock()

    def _draw_overlay(self, frame, latency):
        # 원본 이미지에 직접 텍스트를 그립니다 (가장 확실한 시각화)
        h, w = frame.shape[:2]
        
        # FPS Bars (Visual)
        cam_fps = self.worker.prof_interval.get_fps()
        bar_len = int(min(cam_fps, 60) * 5)
        color = (0, 255, 0) if cam_fps > 25 else (0, 0, 255)
        
        cv2.rectangle(frame, (20, h-40), (20+bar_len, h-20), color, -1)
        cv2.putText(frame, f"CAM: {cam_fps:.1f}", (25, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    def _check_pixel_data(self, frame):
        """프레임이 진짜 검은색인지 데이터 레벨에서 확인"""
        if frame is None: return "None"
        avg_brightness = np.mean(frame)
        shape_str = f"{frame.shape[1]}x{frame.shape[0]}"
        return f"{shape_str} | Brightness: {avg_brightness:.1f}"

    def _update_stats_label(self):
        # 4단계 파이프라인 정밀 분석
        cam_io = self.worker.prof_io.get_avg()       # 카메라 읽기 시간
        cam_int = self.worker.prof_interval.get_avg() # 프레임 간격 (33ms = 30fps)
        tex_up = self.gl_widget.prof_upload.get_avg() # 텍스처 업로드
        gpu_draw = self.gl_widget.prof_draw.get_avg() # 그리기 시간
        
        ui_fps = 1000 / (self.ui_fps_prof.get_avg() + 0.001)

        # 병목 진단 로직 (수정됨: Read 시간은 Blocking이라 무시하고, Interval로 진짜 FPS 측정)
        diagnosis = "✅ SYSTEM OK"
        
        if cam_int > 45: # 22 FPS 미만일 때만 경고
            diagnosis = "❌ SOURCE SLOW (Camera FPS < 22)"
        elif cam_int < 1:
            diagnosis = "⏳ Initializing..."
        elif tex_up > 15:
            diagnosis = "❌ UPLOAD SLOW (PCIe/Bandwidth Issue)"
        elif gpu_draw > 16:
            diagnosis = "❌ RENDER SLOW (Shader/GPU Weak)"

        # [Added] 픽셀 데이터 정보 추가
        pixel_info = self._check_pixel_data(self.gl_widget.current_frame)

        txt = (
            f" [DIAGNOSIS] {diagnosis}\n"
            f" 1. [SOURCE] Camera Read: {cam_io:.2f} ms | Interval: {cam_int:.2f} ms ({1000/(cam_int+0.01):.1f} FPS)\n"
            f" 2. [UPLOAD] CPU->GPU:    {tex_up:.2f} ms (Target < 5ms)\n"
            f" 3. [RENDER] Draw Call:   {gpu_draw:.2f} ms (Target < 16ms)\n"
            f" 4. [UI]     Real FPS:    {ui_fps:.1f} FPS\n"
            f" 5. [DATA]   Frame Info:  {pixel_info}"
        )
        self.lbl_stats.setText(txt)
        
        # [New] 1초마다 콘솔에도 찍기
        now = time.time()
        if now - self.last_console_log_time >= 1.0:
            print("\n" + "="*60)
            print(txt)
            print("="*60)
            self.last_console_log_time = now

def main():
    app = QApplication(sys.argv)
    win = DebugWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()