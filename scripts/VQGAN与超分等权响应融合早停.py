#!/usr/bin/env python3
from 早停基线 import main  # 同进程复用公共早停入口。

if __name__ == "__main__":
    main("vqgan_sr_mean")  # 进程名对应正式方案。
