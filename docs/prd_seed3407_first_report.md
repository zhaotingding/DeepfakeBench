# PRD-203-1 seed3407 首轮核验与阻碍

2026-09-27，实际验证。分支对话：`01a0e12a-41bd-7b81-999d-fd62332d03cb`。

## 结论

正式训练未启动。T1物理BS32在第一次训练前向的第二路CLIP编码触发CUDA OOM，未完成反向或Adam，也未进入BS64测试。失败后未重试、未改batch/精度/模型、未启动T2/T3、未操作容器外进程。有限训练队列与后台完成观察器未安装：当前没有仍在执行或等待自动启动的203实验。

## 分工与去重

- 本批仅T1 VAE→T2 blur→T3 identity，seed3407，有符号残差+concat，从预训练独立初始化。
- 2026-09-27约12:41–12:44，两机配置、.state和输出名称扫描均无3407占用；203无训练进程/训练锁占用。4090实际F1 seed2026运行，F2预留队列未改动，R3不安排。
- 官方App Server读取总对话，确认其已登记203独占3407；读取4090最近对话，与服务器状态核对。全部4090操作只读，保留其用户改动。

## 基线与差异

- 203基线 `b5fb097740a0815f468faac7d6feac402b29e518`，新分支 `prd-203-1-seed3407`。4090核验HEAD `76c7ae0`。
- Trainer、early_stopping、feature_encoders、probe_operators、metrics/utils、dataset/albu的文件SHA-256跨机一致。
- detector的已见差异为离线权重加载和本批不用的替代VAE路径解析；train.py差异含输出路径/唯一时间戳、metadata时点、DDP初始化以及早停开关。本批单卡不走DDP。
- 203原默认关闭早停；3407三份配置显式 `early_stopping.enabled=true,max_declines=3`，与4090固定三轮连续下降规则对齐。其余配置来自4090对应seed2026，仅替换seed/实验标识。
- 保留目标三集平均AUC选checkpoint与epoch末连续下降早停。epoch0评测一次，后续每epoch评测两次；不引入新的选择协议。目标集参与开发选择，不能宣称独立最终测试。
- 203摘要在运行目录，4090在scripts/experiment_summaries；203缺少4090归档器。若恢复启动，仍须适配输出路径、验证归档后再安装队列，不能直接启动现有三份配置后撒手不管。

## 数据、权重与环境

四份JSON SHA-256逐项跨机相同：

| 数据集 | SHA-256 |
|---|---|
| FaceForensics++ | 459fc9891f0bf6e63a1a7d61aed6e96edcc623701d05726b5a225d6fa0cb0b27 |
| Celeb-DF-v2 | 919030ca617e6eb78831fd19470ae2f252ba0452a7497e28014f65fa9c34c29d |
| DFDCP | fc01695baf1f61079edbe6333b4793c10d252523f0ad8341924fa1e1856c11f6 |
| DFDC | ce41a6e936c134f9921747252d666288a0569763df8015d5f7bdeb5272c26796 |

CLIP SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`；VAE SHA-256 `1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5`，均跨机相同。四个LMDB文件大小逐项一致，未做76GB整库hash，不声称数据库逐字节已证同一。

203：Python3.10.21，运行时torch2.5.1+cu124，CUDA12.4，V10032GB；主要ML依赖版本与4090一致，部分辅助包有差异，完整pip清单保存于audit。跨硬件、环境和GPU竞争限制仍在，未来结果差异不能全部归因于seed。

## 本次失败记录

- preflight_id：`prd_seed3407_queue_20260927/smoke_t1`；正式训练run_id：无。
- 开始：2026-09-27 12:46:00 +08:00；结束：12:46:33 +08:00；退出码1。
- screen `4155.prd3407-preflight` 已退出；CUDA报告本次进程PID1604657。正式训练PID：无。
- 日志：`/root/DeepfakeBench-prd-common/.state/prd_seed3407_queue_20260927/smoke_t1.log`。
- 日志SHA-256：`4cf4294dd6ea66a4f9706bfa31c54eac73def274907cf9234edb226c441ac9f4`。
- OOM：总31.73GiB，外部PID2053098占5.45GiB，本次占26.16GiB，剩122.25MiB，申请130MiB失败；25.33GiB由PyTorch实际分配，未分配预留468.07MiB。
- 退出后GPU恢复5582MiB占用、26913MiB空闲；未终止外部进程。不能据此保证外部负载退出后BS32必然能成功。
- 三集AUC、平均AUC、早停轮次、checkpoint路径/hash：均无（首次训练前向失败）。

配置SHA-256：T1 `d995c4809a330949dc5f60a209f8170322686a31992ee17df332d1bfc86ebd86`；T2 `ad0efd6377e65e60b583a7baad3448daf7120682cf16212f4a22d0ae54395351`；T3 `88b300296e11fbde323ef4860137eb8d0d4d9b2f26e1c48a3a9560ab704ac56f`。

## 恢复与汇报边界

本次失败立即通过官方App Server投递总对话，并保存turn/completed回执。未发现可调用的Codex定时任务接口；因启动安全检查失败且无后台实验，不安装闲置轮询器，也不承诺自动重试或未来自动唤醒。PC关闭时不会自动监控/汇报；203也没有依赖PC运行的训练队列。后续若获准恢复，需先解决显存条件、重新完成BS32/64验证、安装并验收有限服务器队列与事件观察器。

待用户决定：等待资源竞争条件改变后重新验证原BS32/64，或明确授权另一种资源/实验条件。降低物理batch、梯度累积、AMP或梯度检查点均未擅自启用。
