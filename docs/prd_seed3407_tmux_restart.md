# 203 tmux完整重训

2026-09-27用户明确授权安装tmux、取消screen训练、完全重新训练。13:10:45终止经cwd/命令/UID核验的旧队列进程组22791，含训练PID22825；全部退出、screen退出、两锁释放，GPU回到5582MiB。未操作外部GPU进程。旧输出保留，取消证据在.state/prd_seed3407_bs16_queue_20260927/cancelled_by_user.json。

安装tmux3.2a，仅新增一个软件包，无升级或卸载。新队列.state/prd_seed3407_bs16_tmux_20260927；任务后缀_203_tmux_20260927；配置保持seed3407/BS16/testBS64，T1→T2→T3，从预训练初始化，不加载旧checkpoint。沿用已通过的同配置BS16/64少量batch验证。

会话prd203-seed3407，train窗口执行队列、logs窗口展示runner.log。SSH断开不停止服务器训练；结束仍以status.json/进程/归档为准，不能只看tmux会话存在。代码语法及真实启动/SSH重连需分别核验。

旧screen结果属于用户主动取消，不得汇入正式三组结果。三目标集参与开发选择，跨4090有seed/batch/环境混杂。中文提交，不push，不碰4090。此文替代此前screen运行状态，旧证据保持原样。
