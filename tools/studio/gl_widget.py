# Project MUSE - gl_widget.py
# OpenGL-based High Performance Viewport (ModernGL + Qt)
# (C) 2025 MUSE Corp. All rights reserved.

import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Slot
import moderngl
import struct
import time

class CameraGLWidget(QOpenGLWidget):
    """
    [High Performance Viewport]
    - Direct Texture Upload (Zero-Copy)
    - Auto Aspect Ratio Corrected
    - Robust Rendering (Pure ModernGL, No QPainter Conflict)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.texture = None
        self.prog = None
        self.vbo = None
        self.vao = None
        
        # 렌더링 상태
        self.frame_width = 0
        self.frame_height = 0
        self.pending_frame = None # [Optimized] 대기 중인 프레임 데이터
        
        # FPS 측정
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_log_time = 0

        # 초기 배경: 검은색
        self.bg_color = (0.0, 0.0, 0.0)

    def initializeGL(self):
        """OpenGL 컨텍스트 및 쉐이더 초기화"""
        print("🎨 [GL] initializeGL() called.")
        try:
            self.ctx = moderngl.create_context()
            print(f"   ✅ [GL] Context Created: {self.ctx.version_code}")
        except Exception as e:
            print(f"❌ [GL] Context Init Failed: {e}")
            return

        # 1. Vertex Shader
        vs = """
        #version 330
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
        """

        # 2. Fragment Shader (BGR -> RGB)
        fs = """
        #version 330
        uniform sampler2D tex;
        in vec2 v_texcoord;
        out vec4 f_color;
        void main() {
            vec4 color = texture(tex, v_texcoord);
            f_color = vec4(color.b, color.g, color.r, 1.0);
        }
        """

        try:
            self.prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        except Exception as e:
            print(f"❌ [GL] Shader Error: {e}")
            return

        # 3. Geometry (Full Screen Quad)
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 1.0, 
             1.0, -1.0, 1.0, 1.0, 
            -1.0,  1.0, 0.0, 0.0, 
             1.0,  1.0, 1.0, 0.0, 
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_vert', 'in_texcoord')])

    def paintGL(self):
        """실제 그리기 (Qt에 의해 호출됨)"""
        if not self.ctx: return

        # [Critical Fix 1] Qt FBO 명시적 바인딩
        # makeCurrent() 없이 호출되므로, 현재 바인딩된 FBO(Qt의 내부 FBO)를 찾아야 합니다.
        try:
            fbo_id = self.defaultFramebufferObject()
            fbo = self.ctx.detect_framebuffer(fbo_id)
            fbo.use()
        except Exception:
            # 초기화 시점 등에서 실패할 수 있음
            return

        # [Critical Fix 2] 텍스처 업로드를 여기서 수행 (Zero-Overhead)
        # render()에서 받은 데이터가 있으면 GPU로 올립니다.
        if self.pending_frame is not None:
            try:
                frame = self.pending_frame
                h, w = frame.shape[:2]

                # 텍스처 생성 (크기 변경 시)
                if self.texture is None or self.frame_width != w or self.frame_height != h:
                    print(f"♻️ [GL] Creating Texture: {w}x{h}")
                    if self.texture: self.texture.release()
                    self.frame_width, self.frame_height = w, h
                    self.texture = self.ctx.texture((w, h), 3, dtype='f1')
                    self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

                # 데이터 전송 (Zero-Copy)
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                self.texture.write(frame)
                self.pending_frame = None # 업로드 완료 처리
            except Exception as e:
                print(f"⚠️ [GL] Texture Upload Error: {e}")

        # FPS 카운트
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
            # 로그 출력 (1초에 한 번)
            if self.frame_width > 0:
                print(f"✨ [GL] Render OK ({self.frame_width}x{self.frame_height}) | FPS: {self.fps}")

        # 1. 뷰포트 계산
        dpr = self.devicePixelRatio()
        w_widget = int(self.width() * dpr)
        h_widget = int(self.height() * dpr)
        
        # 전체 클리어
        self.ctx.viewport = (0, 0, w_widget, h_widget)
        self.ctx.clear(*self.bg_color)

        if self.texture:
            target_ratio = self.frame_width / self.frame_height if self.frame_height > 0 else 16/9
            widget_ratio = w_widget / h_widget if h_widget > 0 else 1

            if widget_ratio > target_ratio:
                view_h = h_widget
                view_w = int(h_widget * target_ratio)
                view_x = int((w_widget - view_w) / 2)
                view_y = 0
            else:
                view_w = w_widget
                view_h = int(w_widget / target_ratio)
                view_x = 0
                view_y = int((h_widget - view_h) / 2)

            try:
                # 텍스처 영역만 그리기
                self.ctx.viewport = (view_x, view_y, view_w, view_h)
                self.texture.use(0)
                self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            except Exception as e:
                pass

    @Slot(object)
    def render(self, frame):
        """메인 스레드 데이터 수신 -> 화면 갱신 요청"""
        # [Optimized] makeCurrent() 제거
        # 비용이 큰 컨텍스트 스위칭 없이 데이터만 넘기고 update() 호출
        if self.ctx is None or frame is None:
            return

        self.pending_frame = frame
        self.update() # -> paintGL() 호출 유도

    def cleanup(self):
        self.makeCurrent()
        try:
            if self.texture: self.texture.release()
            if self.vbo: self.vbo.release()
            if self.vao: self.vao.release()
            if self.prog: self.prog.release()
        except: pass
        finally: self.doneCurrent()