#!/usr/bin/env python3
"""VQGAN 绝对值残差验证早停计算入口。"""
from 早停基线 import main

if __name__ == "__main__":
    main(fixed_variant="vqgan_absolute_residual")  # 复用共用引擎，进程名标明正式方案。
