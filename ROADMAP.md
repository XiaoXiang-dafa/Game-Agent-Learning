# Game Agent Learning Roadmap

> 目标：2025年12月底投递游戏 AI Agent 实习
> 周期：2025年7月中 - 2025年12月底（约5个月）

---

## Phase 1：筑基期（7月中 - 9月初）

**目标**：吃透 Hello-Agents 核心章节，建立 Agent 完整认知框架。

| 周次 | 学习内容 | 产出检查点 |
|------|---------|-----------|
| 1-2 | Ch1-Ch3：Agent 概念、发展史、LLM 基础 | 手绘 Agent Loop 图；会调 Temperature/Top-p |
| 3-4 | **Ch4：手写 ReAct / Plan-and-Solve / Reflection** | GitHub 提交可运行代码，README 记录架构 |
| 5-6 | Ch8（记忆/RAG）+ Ch10（MCP协议） | 给 ReAct 加上 ChromaDB 记忆模块 |

**同步行动**：
- [ ] 注册 GitHub，创建仓库 `game-agent-learning`
- [ ] 每完成一章，commit 一次代码
- [ ] README 持续更新：目前功能、下一步计划、架构图

---

## Phase 2：攻坚期（9月 - 10月底）

**目标**：对接游戏环境，做出可演示的 Demo，这是简历的核心弹药。

| 模块 | 具体任务 |
|------|---------|
| 环境接入 | 对接 MiniGrid（或文字 MUD），状态转文本 Observation |
| 层级规划 | 高层目标 → 子目标 → 原子操作，支持 50+ 步长程任务 |
| 记忆系统 | 短期（滑动窗口）/ 长期（Vector DB）/ 世界知识（RAG） |
| 多 Agent | 双人协作解谜或组队副本（坦克-治疗-输出），MCP 通信 |

**里程碑（10月底）**：
- [ ] Agent 自主通关 MiniGrid 中等难度关卡
- [ ] 录制运行视频/GIF，放 README 首页
- [ ] 发布技术博客一篇

---

## Phase 3：拔高期（11月 - 12月初）

**目标**：项目打磨 + 对齐 JD 加分项。

| 方向 | 行动 |
|------|------|
| 强化学习 | Ch11（GRPO/RLHF），给 Agent 加环境奖励反馈 |
| 自动化测试 | Agent 自主探索地图、遍历任务、记录 Bug 复现路径 |
| 开源贡献 | 给 Hello-Agents 提 1-2 个 PR |
| 简历定稿 | 用"项目经历"填满，零实习但技术深度拉满 |

---

## Phase 4：投递期（12月）

**目标**：把积累转化为面试机会。

- [ ] 内推为主：牛客/脉脉/LinkedIn 找目标公司员工
- [ ] 简历每句话对齐 JD 关键词
- [ ] 准备 3 个核心问题的 3 分钟版本（架构、记忆、规划）
- [ ] Demo 待命：手机能随时展示 GitHub 和运行视频

---

## 每周必做三件事

1. **代码提交**：GitHub 至少 commit 1 次
2. **文档更新**：README 或博客记录本周进展
3. **功能验证**：确保上周代码还能跑

---

## 技术栈

Python · PyTorch · Transformers · LangChain · ChromaDB · MiniGrid
