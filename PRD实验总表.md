# PRD实验总表（共享入口）

科学问题、总进度、分工与下一步的唯一权威版本在[共享分支 codex/prd-progress](https://github.com/John08066/DeepfakeBench/blob/codex/prd-progress/PRD实验总表.md)。本文件仅为入口，不维护分支私有进度表。

训练前先读取远端最新版，两步有依赖，同步失败立即停止：

```bash
git fetch origin codex/prd-progress &&
git show FETCH_HEAD:PRD实验总表.md
```

同时阅读另一张表及项目根目录 `Agent_MCP_Notes.md` 的置顶规则。先登记实验认领并成功push共享分支，才能启动；不合并训练代码，不强推。4090直连不可用时由PC认证Git通过SSH bundle中转，核对远端与服务器ref的SHA，禁止把旧缓存当最新。

2026-10-03初始化共享提交：`fe45d30be4de9c6f6200f53e410caf4605eb0395`。以后以新fetch结果为准。旧 `PROJECT_REPORT.md` 已迁移，原文保留在Git历史。
