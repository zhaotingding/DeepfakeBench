# 4090 双卡实验分工（2026-09-29）

用户已确认服务器两张卡至少在 10 月 7 日前可用，并授权并发实验和独立监控。
为避免超出确定的窗口，新增队列最晚在 2026-10-07 00:00 Asia/Shanghai 停止：
不再派发新组，运行中的新增实验由其独立 timeout 进程组终止，保留日志并标记未完成，不能作为完整结果。

| GPU | 有限顺序队列 | 目的 |
| --- | --- | --- |
| GPU0 | 现有 F1 seed2026 → F2 seed2026；原队列完成后 F1 seed42 → F2 seed42 | 保留现有结果，检验原始/残差单路的 seed 稳健性 |
| GPU1 | R2 abs_diff seed2026 → T1 seed42 → T2 blur seed42 → T3 identity seed42 → R2 abs_diff seed42 | 补齐 seed2026 的残差符号对照，再构成同协议 seed42 配对组 |

- 203-1 已预留 seed3407 的 T1/T2/T3（BS16/64），本批避开，不将硬件/batch差异归因为 seed。
- 4090 本批保持 FF++ c23、训练 BS32 / 测试 BS64、同三测试集、原有早停与选权重规则。seed42 为新种子；R2 seed2026 只相对 T1 seed2026 改 abs_diff 及实验名称。不是从已训练权重接着训练。
- 历史 R2 seed1024 的测试 BS16，与当前 R2 seed2026/42 BS64 不属于完全相同实验。R3 存在维度/容量混杂，仍暂缓，不恢复管理员终止的旧任务。
- 三目标集参与选 checkpoint 和早停，属于开发评估，不宣称独立最终泛化。
- 新实验以配置、seed、训练/测试 BATCHSIZE、probe、residual、feature_mode 对照历史 run_metadata 和203预留清单去重；唯一 task、noclobber、每卡 flock 防止同一任务重复启动。
- GPU0 继续原目录 /home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench；GPU1 使用独立工作树 /home/zhaoting.ding/prd-worktrees/gpu1-20260929，分支 codex/prd4090-gpu1-20260929，独立 Git 索引及结果提交。
- GPU1 的 logs 链接到 NAS 原项目 logs/gpu1_20260929，数据从本地 /home/zhaoting.ding/local_datasets 读取；权重使用原本离线文件，不下载。
- 启动器默认 GPU0，旧两参数调用保持兼容；GPU0 保留原 prd_training.lock，GPU1 独立 prd_training_gpu1.lock。检查占用仅针对目标物理 GPU，不能因为另一卡在训练而拒绝接续。
- 新队列 scripts/run_prd_dual_gpu_queue.sh GPU；每组训练→归档/hash校验→中文 Git 提交→下一组，失败停止。不 push，不覆盖旧输出或无关改动。
- 两卡分别使用现有 prd 与新增 prd4090-gpu1 heartbeat，只预约单次，按实测进度重排并错开；标题 YY-MM-DD HH:mm  PRD4090-GPU0/1进度检查。正常不通知。
- 本文为预留实验矩阵；实际启动、完成与验收以 .state、console.log、run_metadata、GPU PID 和提交为准。有限队列耗尽后先汇总，不盲目追加实验。
