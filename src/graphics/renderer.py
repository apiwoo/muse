# Project MUSE - src/graphics/renderer.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import os
import cv2
import numpy as np
import moderngl
from src.utils.config import Config
from src.utils.logger import get_logger

class Renderer:
    def __init__(self):
        self.logger = get_logger("Graphics_Renderer")
        
        # 1. ModernGL 컨텍스트 생성 (Standalone 모드)
        try:
            self.ctx = moderngl.create_context(standalone=True)
            self.logger.info("🎨 ModernGL Context 생성 완료 (OpenGL)")
        except Exception as e:
            self.logger.error(f"ModernGL 초기화 실패: {e}")
            raise e

        # 2. 렌더링 해상도 설정
        self.width = Config.WIDTH
        self.height = Config.HEIGHT
        
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height), components=3)
        self.fbo.use()

        # 3. 쉐이더 프로그램 컴파일
        self.prog = self._init_shaders()

        # 4. 데이터 로드 (삼각형/선분 인덱스)
        self.indices = self._load_triangulation()
        if self.indices is None:
            self.logger.warning("⚠️ triangulation.npy를 찾을 수 없습니다. Wireframe이 그려지지 않습니다.")
            self.num_indices = 0
        else:
            self.num_indices = len(self.indices)

        # 5. 버퍼 객체 생성
        self.vbo = self.ctx.buffer(reserve=478 * 2 * 4, dynamic=True)
        
        if self.num_indices > 0:
            self.ibo = self.ctx.buffer(self.indices.tobytes())
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, '2f', 'in_vert')],
                self.ibo
            )
        else:
            self.ibo = None
            self.vao = None

        # 배경(카메라 영상) 렌더링용 설정
        self.bg_texture = self.ctx.texture((self.width, self.height), 3)
        self.quad_fs = self._init_quad_shader()
        
        # [FIX] 화면 뒤집힘 해결을 위해 UV 좌표(뒤쪽 2개)를 상하 반전시킴
        # 기존: 0.0, 0.0 (Top-Left 매핑) -> 변경: 0.0, 1.0
        # OpenGL Texture 좌표계와 OpenCV 이미지 메모리 구조 차이 보정
        quad_verts = np.array([
            # x, y, u, v
            -1.0,  1.0, 0.0, 1.0,  # Top Left
            -1.0, -1.0, 0.0, 0.0,  # Bottom Left
             1.0,  1.0, 1.0, 1.0,  # Top Right
             1.0, -1.0, 1.0, 0.0,  # Bottom Right
        ], dtype='f4')
        
        self.quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.quad_fs,
            [(self.quad_vbo, '2f 2f', 'in_vert', 'in_tex')],
        )

        self.logger.info("✨ 렌더링 엔진 준비 완료 (Flip Corrected)")

    def _load_triangulation(self):
        """assets/data/triangulation.npy 파일을 로드합니다."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "../../assets/data/triangulation.npy")
        path = os.path.abspath(path)
        
        if os.path.exists(path):
            data = np.load(path)
            return data.flatten().astype('i4')
        return None

    def _init_shaders(self):
        """얼굴 메쉬를 그릴 쉐이더"""
        vs = """
            #version 330
            in vec2 in_vert;
            
            void main() {
                // MediaPipe (0~1) -> OpenGL (-1~1)
                // [FIX] Y축 반전 해제 (배경과 좌표계 일치시키기 위해 180도 회전)
                // 기존: 1.0 - in_vert.y * 2.0 (Top-Down)
                // 변경: in_vert.y * 2.0 - 1.0 (Bottom-Up)
                gl_Position = vec4(in_vert.x * 2.0 - 1.0, in_vert.y * 2.0 - 1.0, 0.0, 1.0);
            }
        """
        fs = """
            #version 330
            out vec4 f_color;
            void main() {
                // Cyan Color (R, G, B, A)
                f_color = vec4(0.0, 1.0, 1.0, 0.6);
            }
        """
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _init_quad_shader(self):
        """배경 렌더링 쉐이더"""
        vs = """
            #version 330
            in vec2 in_vert;
            in vec2 in_tex;
            out vec2 v_tex;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_tex = in_tex;
            }
        """
        fs = """
            #version 330
            uniform sampler2D tex;
            in vec2 v_tex;
            out vec4 f_color;
            void main() {
                f_color = texture(tex, v_tex);
            }
        """
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def render(self, frame, results):
        if frame is None:
            return None

        # 1. 배경 그리기
        self.bg_texture.write(frame.tobytes())
        self.bg_texture.use(0)
        
        self.fbo.use()
        self.ctx.clear()
        
        # 배경 Quad (Triangle Strip)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # 2. 얼굴 메쉬 그리기
        if results and results.multi_face_landmarks and self.vao:
            face = results.multi_face_landmarks[0]
            
            # VBO 업데이트
            vertices = np.array([(lm.x, lm.y) for lm in face.landmark], dtype='f4')
            self.vbo.write(vertices.tobytes())
            
            # 와이어프레임 (LINES)
            self.vao.render(moderngl.LINES)

        # 3. 결과 다운로드
        data = self.fbo.read(components=3)
        image = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        
        return image