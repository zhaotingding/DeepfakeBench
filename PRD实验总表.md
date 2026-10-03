# PRD实验总表（共享入口）

唯一日常协调入口：[共享分支的精简总表](https://github.com/John08066/DeepfakeBench/blob/codex/prd-progress/PRD实验总表.md)。本文件不保存训练分支私有状态。

首次接入、恢复上下文、新认领或状态发布前同步并读取总表：

```bash
git fetch origin refs/heads/codex/prd-progress:refs/remotes/origin/codex/prd-progress &&
git show refs/remotes/origin/codex/prd-progress:PRD实验总表.md
```

普通监控只检查本任务；不要求同时读取细表。研究查重、复用结果或排障时才按ID查[证据索引](https://github.com/John08066/DeepfakeBench/blob/codex/prd-progress/PRD实验详细记录.md)。认领成功push并读回、现场资源核验通过后方可启动。4090可经PC认证Git/SSH bundle中转并核对远端SHA；缓存不等于最新版本。具体职责见 `Agent_MCP_Notes.md` 精简置顶规则。
