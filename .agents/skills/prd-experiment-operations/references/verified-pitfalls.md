# 已验证经验与边界

仅在对应问题出现时读取。迁移日期2026-10-15；下面保留原验证日期、环境和证据级别，本次没有重跑历史验证。旧任务重试次数、资源窗口、监控ID和批准不构成当前授权。

## 运行中的Bash脚本不能原地覆盖

2026-09-29旧F1训练正常退出后，Bash按原读取偏移继续执行被覆盖的脚本，出现截断命令、退出127，导致归档交接中断。新版本应使用新脚本路径或等旧进程退出；NAS CIFS也可能拒绝替换打开文件。已经恢复并验收的训练结果不应因这段历史失败而重跑。

级别：实际故障；文件替换差异另有隔离模拟验证。证据：4090主仓库 `.state/prd_gpu0_repair_20260929/bash_replace_repro.json`、旧F1的 `completion.recovered` / `finalize.recovered`。

## 跨V100/4090教师重放

Q22原生Q10/VAE与Q12/blur教师在Torch2.5.1+cu124、相同完整权重和相同192实际归一化输入下，默认cuDNN TF32的最大logit误差为 `.02700138/.02609253`。用户当次唯一批准显式禁用cuDNN与matmul TF32后，误差降为 `.0002599954605102539/.00004458427429199219`，仍未达到原协议 `1e-5`。

结论只限该环境和协议：TF32影响误差，但剩余跨硬件误差原因未解决。不能凭软件版本/输入一致判重放通过，也不能自行放宽门槛或扫描设置。两次都在教师重放预检停止，0缓存batch、0学生参数更新；这不是学生方法的科学阴性。

级别：2026-10-10实际GPU与独立CPU核验。证据：`/home/zhaoting.ding/prd-worktrees/gpu0-q22-20261010/scripts/experiment_summaries/q22_t01_20261010_precision_preflight_failed/failure_evidence.json`，结果提交 `e6a79036`；`teacher_smoke/native192_replay.json` 保存actual/expected，详细发布记录在共享Q22专属记录。

## 数据键、选点和随机前向

- LMDB查询保留源JSON原始键；斜杠规范化只用于清单标识，不能无条件改查询键。历史CDFv2/DFDCP的val/test出现精确帧重合，不可作独立验证，也不能据此断言FF++训练泄漏。适用原数据版本，换清单后须按当前证据判断。
- 旧基线使用目标三集选checkpoint/早停，不能作为独立泛化证据。旧T3训练态两次CLIP前向受LoRA dropout影响，identity不保证零残差；判零需控制eval与随机性。旧F1/F2与concat头宽不同，不能忽略容量混杂。

级别：2026-09-26代码核验及2026-10-03已有数据验证；未推广到后续所有模型。证据：相应启动版本的 `training/train.py`、`training/early_stopping.py`、`training/detectors/lora_detector.py`；共享细表P04-D01/D02、`docs/prd/records/N02-D01.md`。

## 调度审批、写锁和收件

登记、真实唤醒和SSH成功是不同事实。先核对被拒的动作及原始回执，区分审批拒绝、工具错误、写锁和回调延迟，不把文档PUSH或删除监控被拒统称为定时注册失败。提示词只保留当前阶段、现行授权与停止条件，旧完整运行指令留在实验记录。

2026-10-13本机 `codex-app-tools` 0.1.6的 `automation_update` 单独使用prompt审批；桌面和对话保存的权限不能仅用全局配置推断。Q24同ID“登记→真实回调→官方暂停→读回PAUSED/next_run_at空”当次已闭环。这是特定版本/一次验收，不是所有工具永久免审或未来准点保证；当前接口应重新读其schema。不要改应用数据库、插件清单或全局审批策略来绕过拒绝。

`active writer`不是Git锁。文件存在或看不到标签不足以识别持有者；可用Restart Manager的持有PID及桌面owner/resume日志辅助定位，但中断或关闭对话须有当前授权和安全接口。身份不明时停止相关动作，不强删锁、不终止训练。发送请求后需目标对话回执才能称收件。

证据：PC `docs/prd/ops/scheduler-approval-20261013.md`、`.state/scheduler-repair-20261013/callback_receipt.json`；2026-10-03/04桌面日志及 `prd-continue-20261003/` 投递记录。前者在PC迁移前工作树提交 `25331c3` 已存在，服务器训练工作树不一定有同名文档。

## 203存储与软链接副本

适用203-1-新 `/root/DeepfakeBench-prd-common` 的相应历史归档：`/root/autodl-tmp`数据盘与repo下 `.state/artifacts` / `.state/backups` 所在系统overlay不同，容量判断须对实际输出路径执行df。`outputs/logs`可能主要是完整state_dict权重，而非日志文本；同盘copy2副本会另占空间。

2026-10-12获批维护将18个完成归档组的43对既定 `.pth/.pt` 在逐字节一致后，把备份路径改成指向保留原件的软链接。这些路径可读，但已不是第二份物理备份；移动/删除 `outputs/logs/mechanism` 原件前须检查别名依赖。旧completion中的backup字段不能证明仍有独立副本，历史余量数字也不能用于当前预算放行。

级别：2026-10-11只读磁盘/代码核验；2026-10-12实际维护验收。证据：203结果提交 `758482f46ff263e513321e7fb005c81775a5c357`，`scripts/experiment_summaries/storage_maintenance_20261012/receipt.json`；PC原存储诊断位于 `C:/Users/john1/.codex/visualizations/2026/09/27/01a0e12a-41bd-7b81-999d-fd62332d03cb/disk_2031_20261011/`。本节仅用于识别历史输入依赖，不授予操作203的权限。
