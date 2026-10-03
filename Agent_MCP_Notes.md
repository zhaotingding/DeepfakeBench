# PRD 全局实验协作规则（置顶，2026-10-03）

> 本节优先于下方旧批次记录；任何旧队列、seed矩阵或“不push”说明不得覆盖用户最新要求。唯一全局进度源为 `origin/codex/prd-progress` 的 `PRD实验总表.md` 与 `PRD实验详细记录.md`，不是各训练分支的旧快照。

- **先同步再实验**：相关任务开始/恢复后先fetch共享分支并读两表；训练前按科学问题、方案及数据协议查重，登记服务器号/GPU、负责人、环境/SSH核验时间及实验占用，中文提交并成功push后才启动。push冲突必须重读认领，不force、不盲目启动。远端不可读写则暂停新训练；4090可由本机认证Git通过SSH bundle中转，不能复制凭据。
- **研究方向**：围绕PRD的探针T、残差r、编码器φ、真假/域分离及可证伪机制推进；不把换seed或常规超参扫点当后续主线。seed/BS只是复现字段；T1=R1=F3，失败重试和同权重评测不计新机制实验。下一候选先诊断幅度/方向与等强度探针，再决定是否训练，未获授权不自行扩展。
- **仅两张权威表**：总表记录科学问题、结论边界、分工及下一步；细表记录服务器/运行状态与结果证据。每次开始、完成、失败、取消或交接精简更新自己相关行；无变化不刷记录。登录情况只记SSH可达、环境和核验时间，不保存密码/token/私钥。旧PROJECT_REPORT.md已迁移并删除，历史由Git保存，不另造第三份进度文档。
- **保护运行与证据**：不覆盖结果、不重复训练、不操作其他用户进程；不得原地改运行中的Bash。目标三测试集参与checkpoint/早停选择的结果称开发评估；完成须训练退出、权重归档hash及结果Git证据。先确认恢复completion的权威性，不据旧失败文件重跑。
- **2026-10-03队列变更**：保留4090当前F1/T1 seed42训练；GPU0后续F2与GPU1后续T2/T3/R2 seed42已写`cancellation.json`取消。旧循环的`set -e`和下一任务`mkdir`防覆盖门槛会在启动前退出；当前组仍先归档、提交。该预期exit1不是需要恢复的训练故障，禁止删除取消标记/重启旧循环。实际收尾待监控验收。资源窗口仍截至2026-10-07 00:00北京时间。
- **Git范围**：中文提交，专业名保留字母。共享两表及入口文档已获授权push；不因此推送各实验分支的代码或历史。监控按进度设置明确日期DTSTART+COUNT=1的单次预约，错开两卡，正常无实质变化安静。

证据：2026-10-03用户要求；两表的运行/归档/提交索引；两卡`.state/prd_dispatch_hold_20261003/decision.json`及取消任务`.state/<task>/cancellation.json`。验证等级：SSH/当前状态与代码路径实际核验；未来自动收尾、未来服务器访问凭据不作保证。

# PRD 实验操作笔记

## 监控频率与顺序队列

- 用户于 2026-09-26 要求：每次检查依据最近完整 epoch 的训练与评测耗时、当前 epoch、连续下降计数和队列交接时间，预估下次检查间隔；禁止默认固定每小时唤醒对话。预计长时间无事件时延长间隔，临近结束或存在异常时缩短；普通状态保持安静。不要把应用 heartbeat 当成免费的服务器检查，它会触发模型调用。
- 初始尚无本次轮耗时时参考历史同配置运行；第一次正式检查再重估。建议长运行阶段约 4–8 小时、结束窗口约 1–2 个 epoch；早停依赖尚未观测的指标，不承诺精确结束时间。
- 已确定的下一实验由服务器队列在上一组正常退出、权重校验归档、摘要 Git 提交后自动启动，无需等待模型定时检查；异常、归档或提交失败必须停队列，不能跳过或重复启动。
- 适用：4090-48G，远程 `/home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench`，GPU0 单卡。唯一训练入口为 `scripts/launch_prd_training.sh`，具有 flock 和日志不覆盖保护；不操作其他用户进程、不 push。
- 证据：本次用户要求；远程启动器和 `.state/prd_t2_blur_seed2026_20260925/run.sh`；本地 `.codex/automations/prd/automation.toml`（旧任务每小时、已暂停）。验证日期 2026-09-26：旧队列正常归档和提交已实际验证；自适应监控策略是用户要求，尚需下一次定时触发验证。

## 实验解释约束

- 当前协议用 CDFv2 / DFDCP / DFDC 三集平均 AUC 选择 checkpoint 和早停，属于目标测试集参与选择的探索结果；不能当作独立泛化评测。
- 早停是完整 epoch 末平均 AUC 连续三次逐轮下降，不是三轮未刷新历史最好；epoch 从 0 开始。
- T3 identity 在训练态对同一图像两次 CLIP 前向，LoRA dropout=0.05，残差未必为零；F1/F2 的分类头宽度与 concat 不同，存在容量混杂。
- 证据：远程 `training/train.py`、`training/early_stopping.py`、`training/detectors/lora_detector.py`；2026-09-26 代码核验，T3 零残差解释不成立的判断基于代码。

## 2026-09-26 配对队列

- 有限队列：T3 identity → F1 orig → F2 residual，seed=2026、训练 BS32、测试 BS64；与已完成的 T1/T2 seed2026 配对。每组从原始预训练权重初始化，不继承上一组训练权重。
- 队列入口 `scripts/run_prd_seed2026_queue.sh`；状态 `.state/prd_seed2026_queue_20260926/runner.log`，每组 `.state/prd_<实验>_seed2026_20260926/completion.json`、`result_commit.txt`；队列完成文件仅在全部归档与提交后写出。队列结束不自行添加新实验。
- GPU0 真实 FF++ 小批量验证：三组各两次前向、反向、Adam 更新，loss 与梯度有限；特征宽度 T3=2048，F1/F2=1024。证据 `.state/prd_seed2026_queue_20260926/smoke.log`；2026-09-26 实际验证，仅证明小批量路径可运行，完整训练待完成。
- 初次监控暂估 6 小时后；后续按本次实际 epoch 耗时重估，不把 6 小时设成永久固定频率。修改应用调度文件后须区分“配置已写入”与“实际定时触发已验证”。

## 2026-09-29 双卡并发

- 用户授权管理员空闲窗口内同时使用两卡；此前“GPU0 单卡、F2 后停止”仅适用于旧批次，现以 scripts/prd_dual_gpu_20260929.md 的有限矩阵为准。GPU0 原 F1→F2 seed2026 完整结束后接 F1→F2 seed42；GPU1 为 R2 seed2026→T1/T2/T3/R2 seed42；避开203-1已预留的seed3407。
- GPU0 保留原目录、分支及 prd_training.lock；GPU1 使用 /home/zhaoting.ding/prd-worktrees/gpu1-20260929、codex/prd4090-gpu1-20260929 和独立 prd_training_gpu1.lock。同用户任意训练检查会误阻挡另一卡，现只检查目标 GPU；不把另一卡训练当重复任务。GPU1 logs 链接至 NAS 原项目 logs/gpu1_20260929，数据保持本地。
- 归档和结果提交在各自工作树进行，避免共用 Git 索引。旧 GPU0 队列运行中，不修改其已加载的循环；新增 GPU0 等待队列只在旧 completion.json 完整成功后开始。新增队列的资源边界为 2026-10-07 00:00 Asia/Shanghai，timeout 只针对自己的新增训练；窗口结束保留未完成状态，不冒充完整实验结果。
- 两个监控 ID：GPU0=prd；GPU1=26-09-29-17-30-prd4090-gpu1。标题 YY-MM-DD HH:mm  PRD4090-GPU0/1进度检查，独立单次时间、错开、按实测重估。首次已登记 2026-09-29 16:30 / 17:30，实际触发待验证。
- 证据：准备提交 e8bd3ba、.state/prd_dual_gpu_setup_20260929/prepared_manifest.json、GPU1 .state/prd_dual_gpu_preflight_20260929/smoke.log、两卡队列日志；2026-09-29 实际验证 R2 BS32 两步反向/Adam、三测试集 BS64 前向通过；GPU0 原 PID3177316 与 GPU1 正式 PID692711 同时运行。后续 PID 和实验结果必须重新核验。

## 运行中 Bash 文件的修改约束（2026-09-29）

- F1 seed2026 在 16:15 正常早停并写出 TRAINING_EXIT_CODE=0；早前原地重写正在等待 Python 的 Bash 启动器，使其恢复读取时出现截断命令 ome/.../python，外层误报127并阻断归档。这是启动器更新引起的交接故障，不是模型训练失败。
- 禁止原地覆盖仍可能被 Bash 读取的脚本。本机 ext4 的隔离模拟中，原子替换可保留运行进程的原脚本；本 NAS CIFS 挂载对打开脚本的 rename 返回 Permission denied，因此运行期间应保留原文件，采用不同文件名的新脚本供后续启动，或等读取进程全部退出后再替换。不能把写文件成功当作运行中更新安全。
- F1 原 completion.json/finalize.log 原样保留；实际正常完成和校验归档以 completion.recovered.json、finalize.recovered.log、result_commit.txt 为准。scripts/recover_prd_gpu0_20260929.sh 只接续未启动的 F2 seed2026，随后进入 prd_gpu0_queue_resumed_20260929 的 F1/F2 seed42。旧失败日志不能触发重跑 F1。
- 证据：旧 runner.log、.state/prd_gpu0_repair_20260929/bash_replace_repro.json、F1恢复归档manifest（SHA256=7a46f4fa311733aea80ca862ab3c81c8638128d6c3a5b03c9609e4f068d8e505）。F1结束及归档为实际验证；文件替换差异为隔离模拟验证；后续实验状态仍须实时核验。
