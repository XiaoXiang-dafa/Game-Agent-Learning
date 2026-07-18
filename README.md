# Game Agent Learning

Experimental framework for LLM-driven game agents: ReAct, memory systems, and multi-agent collaboration.

## 项目简介

面向复杂游戏场景，构建基于大语言模型的 Agent 认知架构实验框架。

核心探索方向：
- 🧠 **感知-规划-行动闭环**：基于 ReAct 范式，实现 Agent 与游戏环境的动态交互
- 🗂️ **分层记忆系统**：短期工作记忆 / 长期经验记忆（Vector DB）/ 世界知识记忆（RAG）
- 🤝 **多 Agent 协作**：基于 MCP 协议的角色分工与通信机制
- 🎮 **游戏环境对接**：已计划对接 MiniGrid 等沙盒环境，验证长程任务规划能力

## 学习路线

| 阶段 | 时间 | 目标 |
|------|------|------|
| Phase 1：筑基 | 7-9月 | 吃透 Hello-Agents 核心章节，手写 ReAct + 记忆系统 |
| Phase 2：攻坚 | 9-10月 | 对接 MiniGrid，实现层级规划与多 Agent 协作 Demo |
| Phase 3：拔高 | 11-12月 | RL 优化 + 自动化测试 + 开源贡献 |
| Phase 4：投递 | 12月 | 简历定稿，集中投递游戏 AI Agent 实习 |

## 技术栈

Python · PyTorch · Transformers · LangChain · ChromaDB · MiniGrid

## 致谢

学习参考 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 开源教程。
