# Project MUSE - gl_widget.py
# OpenGL-based High Performance Viewport (CUDA Interop Ready)
# (C) 2025 MUSE Corp. All rights reserved.

import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Slot
import moderngl
import cv2

# [GPU Support Check]
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

class CameraGLWidget(QOpenGLWidget):
    """
    [High Performance Viewport v2.0]
    - CUDA PBO(Pixel Buffer Object) 매핑을 통한 Zero-Copy 렌더링 지원 시도
    - Fallback: Pinned Memory를 이용한 고속 CPU->GPU 업로드
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.texture = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.pbo = None # Pixel Buffer Object for async upload
        
        # 렌더링 상태
        self.frame_width = 0
        self.frame_height = 0
        self.pending_frame = None 
        self.bg_color = (0.0, 0.0, 0.0)
        
        # Pinned Memory (CPU 측 버퍼, 복사 속도 향상용)
        self.pinned_mem = None

    def initializeGL(self):
        """OpenGL 컨텍스트 및 쉐이더 초기화"""
        print("[GL] initializeGL() called.")
        try:
            self.ctx = moderngl.create_context()
            print(f"   [OK] [GL] Context Created: {self.ctx.version_code}")
        except Exception as e:
            print(f"[ERROR] [GL] Context Init Failed: {e}")
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

        # 2. Fragment Shader (BGR -> RGB Swizzle in Shader)
        # 텍스처 업로드 시 변환하지 않고 쉐이더에서 처리하여 CPU 부하 감소
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
            print(f"[ERROR] [GL] Shader Error: {e}")
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

    def resizeGL(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

    def paintGL(self):
        """실제 그리기 (Qt에 의해 호출됨)"""
        if not self.ctx: return

        # [Critical Fix 1] Qt FBO 명시적 바인딩 (검은 화면 해결 핵심)
        try:
            fbo_id = self.defaultFramebufferObject()
            fbo = self.ctx.detect_framebuffer(fbo_id)
            fbo.use()
        except Exception:
            return

        # [Critical Fix 2] 텍스처 업로드 로직 (GPU/CPU Hybrid)
        if self.pending_frame is not None:
            self._upload_texture(self.pending_frame)
            self.pending_frame = None # 처리 완료

        # Viewport Setup
        dpr = self.devicePixelRatio()
        w_widget = int(self.width() * dpr)
        h_widget = int(self.height() * dpr)
        
        self.ctx.viewport = (0, 0, w_widget, h_widget)
        self.ctx.clear(*self.bg_color)

        if self.texture:
            self._draw_texture(w_widget, h_widget)

    def _upload_texture(self, frame):
        """
        프레임 타입(CuPy/Numpy)에 따라 최적의 업로드 방식 선택
        """
        try:
            # 1. 메타데이터 확인
            if hasattr(frame, 'shape'):
                h, w = frame.shape[:2]
            else:
                return

            # 2. 텍스처 초기화 (크기가 다르면 재생성)
            if self.texture is None or self.frame_width != w or self.frame_height != h:
                if self.texture: self.texture.release()
                self.frame_width, self.frame_height = w, h
                # RGB=3, BGR=3. 쉐이더에서 스위즐링하므로 그대로 업로드
                self.texture = self.ctx.texture((w, h), 3, dtype='f1')
                self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                
                # PBO 생성 (비동기 업로드용)
                # self.pbo = self.ctx.buffer(reserve=w * h * 3)

            # 3. 데이터 전송
            if HAS_CUDA and hasattr(frame, 'device'):
                # [GPU Path] CuPy Array
                # 이상적으로는 CUDA-GL Interop을 써야 하지만, PyCUDA 없이 복잡함.
                # 차선책: VRAM -> Pinned Memory -> Texture (Fast Copy)
                # 일반 .get()보다 빠름
                
                # Pinned Memory가 없거나 크기가 다르면 할당
                nbytes = frame.nbytes
                if self.pinned_mem is None or self.pinned_mem.nbytes != nbytes:
                    self.pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
                
                # VRAM -> Pinned RAM (Async possible)
                frame.get(out=np.frombuffer(self.pinned_mem, frame.dtype, frame.size).reshape(frame.shape))
                
                # Pinned RAM -> OpenGL Texture
                self.texture.write(self.pinned_mem)
                
            else:
                # [CPU Path] Numpy Array
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                self.texture.write(frame)
                
        except Exception as e:
            print(f"[WARNING] [GL] Upload Error: {e}")

    def _draw_texture(self, w_widget, h_widget):
        # Aspect Ratio Correction (Letterboxing)
        target_ratio = self.frame_width / self.frame_height if self.frame_height > 0 else 1.77
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
            self.ctx.viewport = (view_x, view_y, view_w, view_h)
            self.texture.use(0)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
        except: pass

    @Slot(object)
    def render(self, frame):
        """
        메인 스레드 데이터 수신 -> 화면 갱신 요청
        frame: Numpy array or CuPy array
        """
        if self.ctx is None or frame is None:
            return

        # 이전 프레임이 처리되지 않았으면 스킵 (Drop Frame logic for UI responsiveness)
        # if self.pending_frame is not None:
        #     return 

        self.pending_frame = frame
        self.update() # -> paintGL() 호출

    def cleanup(self):
        self.makeCurrent()
        try:
            if self.texture: self.texture.release()
            if self.vbo: self.vbo.release()
            if self.vao: self.vao.release()
            if self.prog: self.prog.release()
            if self.pbo: self.pbo.release()
        except: pass
        finally: self.doneCurrent()