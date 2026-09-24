# 本机三宿主环境

运行位置是 Windows 当前电脑的 WSL2 `Ubuntu`，通过发行版内的 Docker Engine + Compose 运行。每个宿主使用独立镜像；Windows 全局 Python 不安装这些框架依赖。适用范围是本仓库的 ARK、Agent Laboratory、EvoScientist 无界面实验入口。

| 宿主 | 本地镜像 | 主要环境 |
|---|---|---|
| ARK | `rac-local/ark:validated` | Python 3.12、Conda `ark-base`、OpenHands 1.16.0 独立工具环境、PaperBanana、LaTeX、Pandoc、tmux |
| Agent Laboratory | `rac-local/agent-laboratory:validated` | Python 3.11、PyTorch 2.5.1+cpu、TensorFlow CPU 2.18、sentence-transformers 3.3.1、原仓库固定依赖、LaTeX |
| EvoScientist | `rac-local/evo-scientist:validated` | Python 3.13、上游 uv.lock 依赖、DeepAgents/LangGraph、QuickJS、NumPy/SciPy/Pandas/Matplotlib、Node/npm、LaTeX/latexmk/Pandoc |

Agent Laboratory 的 MiniLM 模型和 cl100k_base 分词器已经缓存。它的宿主默认使用离线 Hugging Face 缓存；需要额外模型时可显式覆盖 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE`。CORE/Discovery 的任务容器拥有自己的依赖环境，不继承此设置。

ARK 的完整源码和论文 venue 模板通过校验后的源码归档装入镜像，避免被上游 `.dockerignore` 排除；PaperBanana 固定子模块另行校验。`ark-base` 已配置宿主/RAC 源码路径，Conda Python 可以直接导入这些包。

这是 CPU 环境。CORE capsule 和 Discovery 任务的特有依赖仍在各自 episode 内安装；不能预先灌入 CORE 的干净任务镜像。语音、消息渠道、云端集群部署和独立网页应用等可选产品功能不属于当前 benchmark 宿主运行依赖。

## Windows 使用

在仓库根目录的 PowerShell 执行：

```powershell
.\scripts\hosts.ps1 -Action status
.\scripts\hosts.ps1 -Action check
.\scripts\hosts.ps1 -Action shell -HostName ark
.\scripts\hosts.ps1 -Action run -HostName agent_laboratory -HostArguments @('doctor-host', '--host', 'agent_laboratory')
```

`check` 不调用模型；以断网容器检查锁定源码、`pip check`、完整宿主模块导入、LaTeX 编译、原生任务执行 hook、Agent Laboratory 的真实离线向量计算，以及 ARK 的 Conda/OpenHands/PaperBanana。另验证任务容器只有 bootstrap 包、文件同步、超时杀子进程、禁止复用和退出清理。hook 检查使用测试执行端，不能代替真实 LLM benchmark 验收。

报告写入 `runs/environment-check/<UTC时间>/`，包括 `summary.json`、每宿主日志和完整 `pip freeze`。报告保存实际镜像 ID；正式实验应固定该 ID，不能只依赖可变 tag。

## 从源码重新配置

本机已完成安装，无需重复。新机器或重建时：

1. 使用 Python 3.10+ 执行 `python scripts/bootstrap_host_archives.py`，下载并核对三个固定源码和 PaperBanana。已有目录只验证，不覆盖。
2. 执行 `python scripts/cache_host_models.py`，下载固定模型、分词器和 Miniforge，并校验安装器官方 SHA256。当前电脑的 Windows 网络可下载这些资产，WSL 直接访问部分源不稳定，建议在 Windows 执行此步。
3. 在 Ubuntu 以 root 执行 `bash scripts/setup_linux_docker.sh`。它使用 Docker 官方 apt 仓库，不安装 Docker Desktop。
4. PowerShell 执行 `.\scripts\hosts.ps1 -Action build`。本地 Compose 覆盖使用 AWS Public ECR 上的官方 Docker Library Python 副本，避开本机不可用的 Docker Hub 连接。
5. 构建任务容器并验收：

```powershell
wsl -d Ubuntu -u root --cd C:\Users\Leo\RAC-AI_Scientist -- docker build --build-arg PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.11-slim -f docker/Dockerfile.task-runtime -t rac-local/task-runtime:validated .
.\scripts\hosts.ps1 -Action check
```

仓库路径或 WSL 发行版名称变化时替换示例参数；启动脚本支持 `-Distribution`。三镜像的 Python/依赖互相隔离，磁盘占用应以 `docker system df` 和 Windows 磁盘剩余容量共同判断。

## 模型和正式实验

此次环境检查没有调用付费模型，也没有产生新的 benchmark 实验结果。正式执行前仍需在被 Git 忽略的 `.env` 配置 `AGENT_API_BASE`、`AGENT_API_KEY`、`AGENT_MODEL_NAME`；评分器另需 judge 配置。R1–R3 每个 episode 需要独立 SharedNet 房间，N0 不需要。

模型调用、真实任务输出与外部评分的验收步骤见 [benchmark 适配说明](benchmark-adapters.zh-CN.md)。任务 runtime 由 WSL controller 管理，宿主只接收 endpoint/token，不给宿主挂 Docker socket。

本地 Compose 已把 `host.docker.internal` 指向 Docker host，并传递 `RAC_TASK_RUNTIME_URL` / `RAC_TASK_RUNTIME_TOKEN`。例如 controller 监听受信任实验网络的 8765 端口时，可使用 `http://host.docker.internal:8765`。必须额外保证 workspace 在 controller、宿主和任务容器中的绝对路径一致；仅设置 URL 不足以启动 CORE episode。
