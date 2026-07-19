# Game Agent Learning

面向游戏场景的大语言模型（LLM）AI Agent 学习与实验项目，系统性地探索 Agent 认知架构、长程规划、记忆系统、多 Agent 协作与强化学习优化等核心课题。

## 研究背景

以 LLM 为认知核心的 AI Agent 在软件开发、科学研究、办公自动化等通用场景中已展现出强大潜力。游戏作为具备明确目标、丰富交互、复杂规则与动态环境的"天然 Agent 试验场"，对 Agent 在**实时决策、长程规划、多角色协作、环境泛化**等方面提出了远超通用场景的挑战。当前基于 LLM 的 Agent 架构在游戏领域的适配与优化仍处于早期阶段。

本项目围绕以下典型应用场景开展实验：
- 🎯 **自主任务通关**：Agent 理解任务目标，感知游戏画面，调用操作接口，完成副本通关、主线推进等长程任务
- 🧑‍🤝‍🧑 **智能 NPC**：LLM 驱动的 NPC，具备个性化性格，根据玩家行为动态调整对话与行为
- 🧪 **自动化测试**：Agent 自主探索地图、遍历任务流程、发现并复现 Bug

## 核心探索方向

对应游戏 AI Agent 的五大核心挑战：

| # | 挑战 | 探索内容 |
|---|------|----------|
| 1 | **Agent 认知架构** | 设计模块化的感知-记忆-规划-行动-反思架构，标准化游戏环境接口与观测-动作抽象层 |
| 2 | **长程规划与层级决策** | 高层策略→子目标→操作序列的层级分解，LLM 推理 + RL 策略优化的动态重规划 |
| 3 | **记忆系统设计** | 短期工作记忆（任务上下文）+ 长期经验记忆（向量检索）+ 世界知识记忆（RAG） |
| 4 | **多 Agent 协作与博弈** | 多 Agent 通信协议、角色分工、不完全信息下的团队策略涌现 |
| 5 | **奖励系统与 RL 优化** | 稠密/稀疏奖励信号设计，LLM 自评估 + 过程奖励模型（PRM），PPO/GRPO 策略优化 |

## 项目结构

```
Game-Agent-Learning/
├── chapter1/                  # ✅ 已完成
│   ├── travel_agent.py        #   基于 ReAct 的旅行助手 Agent
│   ├── requirements.txt       #   Python 依赖
│   └── README.md              #   章节说明
├── chapter2/                  # ✅ 已完成
│   ├── Transformer.py         #   从零实现的 Transformer 架构
│   └── README.md              #   章节说明
└── README.md
```

### Chapter 1：初识智能体 ✅

实现了一个基于 **ReAct 范式**（Thought → Action → Observation 循环）的智能旅行助手，覆盖：

- **工具调用**：实时天气查询（wttr.in）+ 天气预报（Open-Meteo，±10 天）+ 景点推荐（Tavily Search）
- **Prompt Engineering**：结构化 system prompt 驱动 LLM 按格式输出
- **主循环解析**：正则提取 Action → 动态调用工具 → 追加 Observation

→ 详见 [chapter1/README.md](chapter1/README.md)

### Chapter 2：Transformer 从零实现 ✅

基于 PyTorch 逐行手写 Transformer 架构，对应论文 "Attention Is All You Need"，覆盖：

- **位置编码**：sin/cos 为序列每个位置生成唯一编码
- **多头注意力**：核心模块，拆成多个头各自做缩放点积注意力
- **Encoder-Decoder**：完整堆叠结构，含残差连接与 LayerNorm
- **Mask 机制**：padding mask + 下三角 subsequent mask

→ 详见 [chapter2/README.md](chapter2/README.md)

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | ReAct, Tool Use, Prompt Engineering |
| LLM 推理 | OpenAI 兼容 API（支持任意模型） |
| 深度学习 | PyTorch |
| 信息检索 | Tavily Search API |
| 天气数据 | wttr.in / Open-Meteo |
| 语言 | Python 3.10+ |

## 环境搭建

```bash
# 克隆仓库
git clone git@github.com:XiaoXiang-dafa/Game-Agent-Learning.git
cd Game-Agent-Learning

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r chapter1/requirements.txt
```

在项目根目录创建 `.env` 文件（已通过 `.gitignore` 排除）：
```bash
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o
TAVILY_API_KEY=your_tavily_key
```

## 致谢

学习参考 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 开源教程。