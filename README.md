# WebClaw DeepAgent

> 一个基于 LangGraph + Playwright + MCP 的 Web 智能体系统

## 项目简介

WebClaw DeepAgent 是一个能够执行复杂 Web 任务的智能 Agent 系统。它融合了：
- **LangGraph**：多代理编排框架
- **Playwright**：浏览器自动化
- **MCP**：工具协议标准（后续阶段）
- **DeepAgent**：深度规划能力（后续阶段）

## Week 1 功能

当前实现了最小可用原型（MVP）：

1. ✅ 基础浏览器控制（打开页面、点击、输入、滚动）
2. ✅ 截图功能
3. ✅ 文本提取
4. ✅ 单代理执行器（ReAct 模式）
5. ✅ LLM 决策驱动

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行测试

```bash
# 测试浏览器工具
python tests/test_week1.py

# 运行一个简单任务
python run.py --task "打开百度搜索Python" --model deepseek-chat
```

### 4. 运行 Agent

```bash
# 可视模式（能看到浏览器窗口）
python run.py --task "打开豆瓣电影Top250，找到评分最高的电影名称"

# 无头模式（后台运行）
python run.py --task "your task" --headless
```

## 项目结构

```
webclaw-agent/
├── agents/                 # 代理模块
│   ├── executor.py         # 执行器代理（Week 1）
│   ├── state.py            # 状态定义
│   └── graph.py            # LangGraph 主图（Week 2）
│
├── tools/                  # 工具模块
│   └ playwright_browser.py # Playwright 浏览器工具
│   └ mcp_tools.py          # MCP 工具（Week 2）
│
├── config/                 # 配置
│   └ config.yaml           # 主配置文件
│
├── tests/                  # 测试
│   └ test_week1.py         # Week 1 测试脚本
│
├── logs/                   # 日志目录
├── screenshots/            # 截图保存目录
│
├── run.py                  # 主运行入口
├── requirements.txt        # 依赖列表
└ .env.example              # API Key 配置模板
└ README.md                 # 本文件
```

## 支持的模型

| 模型 | API Key 变量 | 费用参考 |
|------|-------------|---------|
| DeepSeek Chat | DEEPSEEK_API_KEY | ~0.01元/千token |
| Qwen Plus | QWEN_API_KEY | ~0.02元/千token |
| GPT-4o | OPENAI_API_KEY | ~0.15元/千token |
| GPT-4o-mini | OPENAI_API_KEY | ~0.01元/千token |

## Week 1 测试标准

运行以下命令验证 Week 1 完成：

```bash
python run.py --task "打开 https://www.baidu.com，搜索'Python'，截图保存"
```

预期结果：
- ✅ 浏览器打开百度
- ✅ 输入框填入 "Python"
- ✅ 点击搜索按钮
- ✅ 截图保存到 `screenshots/` 目录

## 下一步计划

- **Week 2**：增加 ReAct 推理 + 多工具调用
- **Week 3**：多代理编排（Planner + Executor）
- **Week 4**：MVP 集成测试，成功率目标 ≥50%

## 开发进度

- [x] Week 1: 项目初始化 + 单代理骨架 ✅
- [ ] Week 2: ReAct 推理 + 工具调用
- [ ] Week 3: 多代理编排
- [ ] Week 4: MVP 集成测试

## 许可证

MIT License