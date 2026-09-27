# 203 seed3407 BS16授权修订

2026-09-27用户在BS32 OOM后明确允许203降低BATCHSIZE，并要求独立203分支。
仅203分支prd-203-1-seed3407：T1→T2→T3，物理训练BS16，测试BS64，不使用梯度累积、AMP或新模块；其余学习率/epoch/早停/目标集选择与seed2026探索协议相同。
因此跨4090比较同时混杂seed、batch、硬件/环境。目标三集参与checkpoint选择，不是独立最终测试。BS32失败历史保留于7200cc0；此前“等待用户决定”已被本次明确授权取代。

服务器有限队列scripts/run_prd_203_seed3407.sh使用独立flock，每次调用既有启动器取得全局训练锁。从预训练重新初始化；失败停止；每组正常结束后校验三集指标、复制并SHA256复核平均最佳checkpoint、归档摘要、中文Git提交，才启动下一组；共三组，不自动追加或重跑，不push。
状态.state/prd_seed3407_bs16_queue_20260927/status.json、events.jsonl、runner.pid、runner.log；各组.state/<task>/started.json、completion.json、result_commit.txt。输出由PRD_OUTPUT_ROOT解析。

完整训练验收仍以真实运行结果为准；归档器合成测试不证明真实完成归档。后台screen不依赖本地PC；Codex监控与投递依赖本机在线且App运行，自动触发须与配置保存分别陈述。
