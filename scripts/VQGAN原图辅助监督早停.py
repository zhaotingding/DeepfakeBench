#!/usr/bin/env python3
"""VQGAN原图辅助监督：有符号联合主头加原图线性辅助头，最终只用主头。"""
from 早停基线 import main

if __name__ == "__main__":
    main("vqgan_original_aux")
