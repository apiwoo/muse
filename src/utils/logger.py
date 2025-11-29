# Project MUSE - src/utils/logger.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import logging

def get_logger(name):
    """
    간단한 콘솔 로거를 생성하여 반환합니다.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # 포맷 설정
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger