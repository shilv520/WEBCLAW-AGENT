# WebClaw DeepAgent 项目开发规划（详细版）

> 目标：从0到1构建一个融合 OpenClaw + LangGraph + MCP + Playwright 的Web智能体系统

---

## 项目核心定位

- **本质**：一个能执行复杂Web任务的智能Agent系统
- **技术栈**：LangGraph多代理 + Playwright浏览器控制 + MCP工具协议 + DeepAgent深度规划
- **验证方式**：用5万条真实Web任务测试成功率
- **最终产出**：开源系统 + Web Demo + 可投顶会的论文

---

## 为什么这个规划是正确的？

数据不是第一步。**系统先跑起来，再用数据验证**。

正确的顺序是：
1. 搭骨架（让Agent能执行一个简单Web任务）
2. 迭代优化（增加复杂功能）
3. 大规模测试（用你的5万数据验证成功率）
4. 算法创新（做别人没做过的事）
5. 论文输出

---

## 第一阶段：最小可用原型（Week 1-4）

### 目标
完成一个能跑的Web Agent，能执行"打开网页 → 搜索 → 提取信息"这类简单任务。

### 需要学习的技术
| 技术 | 学习资源 | 用途 |
|------|---------|------|
| LangGraph基础 | LangGraph官方文档 | 构建Agent状态机 |
| Playwright Python | Playwright官方教程 | 浏览器自动化 |
| MCP协议基础 | Anthropic MCP文档 | 工具封装标准 |
| Python Asyncio | Python并发编程 | 异步执行 |

---

### Week 1：项目初始化 + 单代理骨架（已完成代码生成）

#### 步骤 1.1：创建项目目录结构

**已创建的目录结构**：

```
webclaw-agent/
├── agents/                     # 代理模块
│   ├── __init__.py             # 模块初始化
│   ├── executor.py             # 执行器代理（核心）
│   └── state.py                # Agent状态定义
│
├── tools/                      # 工具模块
│   ├── __init__.py             # 模块初始化
│   └── playwright_browser.py   # Playwright浏览器工具
│
├── config/                     # 配置文件
│   └── config.yaml             # 主配置（LLM、浏览器、Agent参数）
│
├── utils/                      # 工具函数（后续扩展）
│   └── __init__.py
│
├── tests/                      # 测试脚本
│   ├── __init__.py
│   └── test_week1.py           # Week 1功能测试
│
├── logs/                       # 运行日志目录
│   └── .gitkeep                # 占位文件
│
├── screenshots/                # 截图保存目录
│   └── .gitkeep                # 占位文件
│
├── states/                     # 状态存储目录
│   └── .gitkeep                # 占位文件
│
├── run.py                      # 主运行入口
├── requirements.txt            # Python依赖列表
├── .env.example                # API Key配置模板
└── README.md                   # 项目说明文档
```

**每个文件的作用**：

| 文件 | 作用 | 是否已创建 |
|------|------|-----------|
| `agents/state.py` | 定义AgentState数据结构，管理任务执行状态 | ✅ |
| `agents/executor.py` | 执行器代理核心逻辑，LLM决策+Playwright执行 | ✅ |
| `tools/playwright_browser.py` | Playwright浏览器封装，提供open/click/type等API | ✅ |
| `config/config.yaml` | LLM模型、浏览器、Agent参数配置 | ✅ |
| `run.py` | 命令行入口，解析参数并启动Agent | ✅ |
| `tests/test_week1.py` | 自动化测试脚本，验证浏览器工具功能 | ✅ |

---

#### 步骤 1.2：安装依赖

**requirements.txt 内容（已创建）**：

```txt
# Core Agent Framework
langgraph>=0.2.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0

# Browser Automation
playwright>=1.40.0

# Data Models
pydantic>=2.0.0

# Async Support
asyncio-throttle>=1.0.0

# Image Processing
pillow>=10.0.0

# Logging
loguru>=0.7.0

# Environment
python-dotenv>=1.0.0

# API (for later phases)
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

**安装命令**：

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 安装Playwright浏览器
playwright install chromium

# 3. 验证安装成功
python -c "import langchain; import playwright; print('安装成功')"
```

**可能遇到的问题**：

| 问题 | 解决方案 |
|------|---------|
| playwright install 失败 | 确保网络通畅，或使用镜像：`playwright install chromium --mirror https://npmmirror.com/mirrors/playwright` |
| langchain版本冲突 | 使用 `pip install --upgrade langchain langchain-openai` |
| Windows缺少Visual C++ | 安装 Microsoft Visual C++ Redistributable |

---

#### 步骤 1.3：配置API Key

**.env.example 内容（已创建）**：

```bash
# DeepSeek (推荐，性价比高)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Qwen (阿里云)
QWEN_API_KEY=your_qwen_api_key_here

# OpenAI (GPT模型)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude模型)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**配置步骤**：

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑.env文件，填入你的API Key
# 推荐：DeepSeek（费用低，效果好）
# API Key申请地址：https://platform.deepseek.com/
```

**各模型费用对比**：

| 模型 | 费用（人民币/千token） | 申请地址 |
|------|----------------------|---------|
| DeepSeek Chat | ~0.01元 | https://platform.deepseek.com/ |
| Qwen Plus | ~0.02元 | https://dashscope.aliyun.com/ |
| GPT-4o-mini | ~0.01元 | https://platform.openai.com/ |
| GPT-4o | ~0.15元 | https://platform.openai.com/ |

---

#### 步骤 1.4：理解核心代码

**AgentState（agents/state.py）核心结构**：

```python
class AgentState(TypedDict):
    task: str                     # 用户任务
    task_id: str                  # 任务ID
    current_step: int             # 当前步骤
    max_steps: int                # 最大步骤
    step_history: List[dict]      # 步骤历史
    browser_state: BrowserState   # 浏览器状态
    thought: str                  # Agent思考
    action: str                   # 执行动作
    action_result: str            # 动作结果
    final_answer: str             # 最终答案
    success: bool                 # 是否成功
    error_message: str            # 错误信息
```

**PlaywrightBrowser（tools/playwright_browser.py）核心方法**：

```python
class PlaywrightBrowser:
    async def start()           # 启动浏览器
    async def close()           # 关闭浏览器
    async def open_url(url)     # 打开网页
    async def click(selector)   # 点击元素
    async def type_text(selector, text)  # 输入文字
    async def scroll(direction) # 滚动页面
    async def wait(seconds)     # 等待
    async def extract_text(selector)     # 提取文本
    async def screenshot()      # 截图
    async def get_current_state()        # 获取状态
```

**ExecutorAgent（agents/executor.py）工作流程**：

```python
class ExecutorAgent:
    async def run(task):
        # 1. 创建初始状态
        state = create_initial_state(task)
        
        # 2. 启动浏览器
        await browser.start()
        
        # 3. 执行循环
        while step < max_steps:
            # 3.1 LLM决策下一步
            decision = await get_llm_decision(state)
            
            # 3.2 Playwright执行动作
            result = await execute_action(decision)
            
            # 3.3 更新状态
            update_state(state, decision, result)
            
            # 3.4 检查是否完成
            if decision["is_complete"]:
                break
        
        # 4. 关闭浏览器，返回结果
        await browser.close()
        return state
```

---

#### 步骤 1.5：运行测试验证

**测试脚本（tests/test_week1.py）提供两种测试方式**：

**方式1：自动化测试（pytest）**

```bash
# 安装pytest（已包含在requirements.txt）
pip install pytest pytest-asyncio

# 运行所有测试
pytest tests/test_week1.py -v

# 预期输出：
# test_start_browser PASSED
# test_open_url PASSED
# test_screenshot PASSED
# test_extract_text PASSED
# test_type_text PASSED
# test_click PASSED
# test_scroll PASSED
# test_get_state PASSED
```

**方式2：手动测试（可视化）**

```bash
# 运行手动测试，能看到浏览器窗口
python tests/test_week1.py

# 测试内容：
# Test 1: 打开百度
# Test 2: 输入"Python"
# Test 3: 点击搜索
# Test 4: 等待2秒
# Test 5: 截图
# Test 6: 获取当前状态
```

---

#### 步骤 1.6：运行第一个完整任务

**命令行运行**：

```bash
# 基础任务（可视化模式）
python run.py --task "打开百度搜索Python"

# 指定模型
python run.py --task "打开百度搜索Python" --model deepseek-chat

# 无头模式（后台运行）
python run.py --task "打开百度搜索Python" --headless

# 查看详细参数
python run.py --help
```

**预期运行结果**：

```
Starting WebClaw DeepAgent
Task: 打开百度搜索Python
Model: deepseek-chat

==================================================
任务执行结果:
==================================================
任务: 打开百度搜索Python
步骤数: 3
成功: True
答案: 已在百度搜索框输入Python并完成搜索
==================================================

执行步骤:
  步骤1: open_url -> 已打开页面: https://www.baidu.com...
  步骤2: type_text -> 已输入文字 'Python'...
  步骤3: click -> 已点击元素: #su...
```

---

#### Week 1 验收标准

**必须完成的检查项**：

- [ ] `pip install -r requirements.txt` 成功，无报错
- [ ] `playwright install chromium` 成功
- [ ] `.env` 文件已配置API Key
- [ ] `pytest tests/test_week1.py` 全部通过（至少6/8项）
- [ ] `python run.py --task "打开百度"` 能成功执行
- [ ] `screenshots/` 目录下有截图文件生成
- [ ] `logs/` 目录下有日志文件生成

**常见问题排查**：

| 问题 | 检查方法 | 解决方案 |
|------|---------|---------|
| API Key无效 | 检查.env文件内容是否正确 | 重新申请API Key |
| 浏览器启动失败 | 检查playwright是否安装 | `playwright install chromium` |
| 网络请求超时 | 检查网络连接 | 使用代理或等待重试 |
| 元素选择器错误 | 检查网页DOM结构 | 使用更精确的selector |

---

---

### Week 2：增加ReAct推理 + 工具调用（待开发）

#### 步骤 2.1：理解ReAct模式

**ReAct = Reasoning + Acting**

工作流程：
1. **Thought**（思考）：分析当前状态，决定下一步
2. **Action**（行动）：执行具体操作
3. **Observation**（观察）：获取执行结果
4. **循环**直到任务完成

**需要修改的文件**：
- `agents/executor.py`：增加ReAct循环逻辑
- `agents/prompts.py`：新建，定义ReAct提示词模板

---

#### 步骤 2.2：增加更多工具

**需要新增的工具**：

| 工具名称 | 功能 | 文件位置 |
|---------|------|---------|
| `find_element` | 查找页面元素 | tools/playwright_browser.py |
| `get_html` | 获取页面HTML | tools/playwright_browser.py |
| `press_key` | 按键操作 | tools/playwright_browser.py（已有） |
| `execute_js` | 执行JavaScript | tools/playwright_browser.py（已有） |

**新增工具文件**：
- `tools/web_tools.py`：Web相关工具集合
- `tools/data_tools.py`：数据提取工具

---

#### 步骤 2.3：实现工具选择逻辑

**LLM需要根据任务选择合适的工具**：

```python
# 工具注册表
TOOL_REGISTRY = {
    "browser": {
        "open_url": "打开网页",
        "click": "点击元素",
        "type_text": "输入文字",
        "scroll": "滚动页面",
        ...
    },
    "data": {
        "extract": "提取数据",
        "parse": "解析内容",
        ...
    }
}

# LLM决策时会看到工具描述，选择最合适的
```

---

#### Week 2 验收标准

- [ ] Agent能执行"打开豆瓣电影Top250，找到评分最高的电影名称"
- [ ] 能自动选择合适的工具
- [ ] 有完整的Thought→Action→Observation循环
- [ ] 成功率 ≥30%（简单任务）

---

### Week 3：多代理编排（Planner + Executor）（待开发）

#### 步骤 3.1：理解多代理架构

**Planner + Executor 模式**：

```
用户任务 → Planner（规划）→ 步骤列表 → Executor（执行每个步骤）→ 结果
```

**例如任务**："打开亚马逊，搜索Python书，找出最便宜的"

Planner分解：
1. 打开亚马逊网站
2. 在搜索框输入"Python编程书"
3. 查看搜索结果列表
4. 提取所有价格信息
5. 找出最低价格的书名
6. 返回结果

Executor逐个执行每个步骤。

---

#### 步骤 3.2：创建Planner代理

**新建文件**：
- `agents/planner.py`：规划器代理

**Planner核心逻辑**：

```python
class PlannerAgent:
    def plan(self, task: str) -> List[str]:
        """把任务分解成步骤列表"""
        # 使用LLM分解任务
        steps = self.llm.invoke(
            f"把这个任务分解成具体步骤：{task}"
        )
        return parse_steps(steps)
```

---

#### 步骤 3.3：创建LangGraph StateGraph

**新建文件**：
- `agents/graph.py`：LangGraph主图

**Graph结构**：

```python
from langgraph.graph import StateGraph

# 定义节点
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)

# 定义边
graph.add_edge("planner", "executor")
graph.add_edge("executor", END)

# 编译
app = graph.compile()
```

---

#### Week 3 验收标准

- [ ] Planner能正确分解复杂任务
- [ ] Executor能逐个执行步骤
- [ ] LangGraph图能完整运行
- [ ] 成功率 ≥40%（中等复杂任务）

---

### Week 4：MVP集成测试（待开发）

#### 步骤 4.1：整合所有模块

**整合内容**：
- Planner + Executor 协同工作
- 完整的工具调用链
- 日志记录系统
- 截图保存系统

---

#### 步骤 4.2：用真实任务测试

**测试任务（10条）**：

1. 打开百度搜索"Python教程"
2. 打开豆瓣电影Top250，找评分最高的电影
3. 打开知乎，搜索"LangGraph"，找第一个回答
4. 打开GitHub，搜索"playwright"，找star最多的项目
5. ...（你自己准备5条）

---

#### Week 4 验收标准

- [ ] 10条测试任务成功率 ≥50%（5/10通过）
- [ ] 每个任务都有完整日志记录
- [ ] 每个任务都有截图保存
- [ ] 错误任务有明确失败原因

---

---

## 第二阶段：核心功能完善（Week 5-8）

### 目标
增加DeepAgent深度规划能力 + 错误自愈 + Skill系统，让成功率提升到70%。

### 需要学习的技术
| 技术 | 学习资源 | 用途 |
|------|---------|------|
| DeepAgent模式 | LangGraph DeepAgent示例 | 子代理生成、递归规划 |
| Graph-of-Thoughts | 相关论文 | 思考过程有向图 |
| Self-Reflection | DeepSeek论文 | Agent自我纠错 |
| OpenClaw Skills格式 | OpenClaw GitHub | 可复用技能模块 |

### 具体任务清单

#### Week 5：DeepAgent深度规划

**新建文件**：
- `agents/deep_reasoner.py`：深度推理代理

**核心功能**：
- 面对复杂任务能深度思考
- 遇到子任务自动创建新Agent
- 递归分解任务

**测试标准**：
```python
# 能执行这个复杂任务：
agent.run("打开某电商网站，找到某商品，对比3个店铺的价格，选出最便宜的")
# Agent能自动分解为：打开网站 → 搜索 → 提取各店铺价格 → 对比 → 返回结果
```

---

#### Week 6：Critic错误自愈

**新建文件**：
- `agents/critic.py`：批评者代理

**核心功能**：
- 检查执行结果是否正确
- 对比预期输出
- 自动回滚和重试

**测试标准**：
```python
# Agent执行出错时能自动发现并纠正：
# 例如：点击了错误按钮 → Critic发现截图不对 → 自动回滚重试
```

---

#### Week 7：Skills系统

**新建文件**：
- `skills/__init__.py`
- `skills/web_search.md`
- `skills/form_fill.md`
- `skills/data_extract.md`
- `skills/login_flow.md`
- `skills/anti_scraping.md`
- `tools/skill_loader.py`

**Skill格式（Markdown）**：

```markdown
---
name: web_search
description: 在网页上进行搜索
tools: [open_url, type_text, click, extract_text]
---

## 使用步骤

1. 打开目标网站
2. 在搜索框输入关键词
3. 点击搜索按钮
4. 等待结果加载
5. 提取搜索结果

## 适用场景

- 需要在特定网站搜索信息
- 电商商品搜索
- 文章内容搜索

## 示例

任务: "在豆瓣搜索电影《肖申克的救赎》"
执行: web_search(url="https://www.douban.com", query="肖申克的救赎")
```

**测试标准**：
```python
# Skills能被复用：
agent.run("登录某网站并提取数据", skills=["login_flow", "data_extract"])
# 自动加载两个Skill组合执行
```

---

#### Week 8：阶段集成测试

**测试任务（50条复杂任务）**：
- 从你的5万数据中抽取50条
- 运行完整测试
- 记录成功率

**验收标准**：
- [ ] 成功率 ≥70%
- [ ] 失败案例有明确原因分析
- [ ] Skill系统正常工作

---

---

## 第三阶段：算法创新（Week 9-12）

### 目标
实现两个核心创新算法，提升成功率到80%+，为论文做准备。

---

### 创新点1：Adaptive Skill Composition

**核心思想**：Agent根据任务复杂度动态组合Skills，并用执行历史学习最优组合策略。

**新建文件**：
- `algorithms/skill_composer.py`
- `algorithms/reward_calculator.py`

**实现步骤**：

1. **设计Skill组合评分函数**

```python
def calculate_reward(skill_combo, execution_result):
    """计算Skill组合的奖励值"""
    success_rate = execution_result["success"]  # 0或1
    execution_time = execution_result["time"]
    
    # 奖励 = 成功得分 - 时间惩罚
    reward = success_rate * 10 - execution_time / 60
    
    return reward
```

2. **实现基于历史的Skill推荐**

```python
class SkillComposer:
    def recommend_skills(self, task: str) -> List[str]:
        """根据历史数据推荐最优Skill组合"""
        # 1. 分析任务特征
        task_features = self.analyze_task(task)
        
        # 2. 从历史数据找相似任务
        similar_tasks = self.history.find_similar(task_features)
        
        # 3. 返回历史最优组合
        best_combo = self.get_best_combo(similar_tasks)
        
        return best_combo
```

3. **实现动态Skill组合生成**

```python
def generate_dynamic_combo(self, task: str, available_skills: List[str]) -> List[str]:
    """动态生成新的Skill组合"""
    # 使用LLM分析任务需求
    analysis = self.llm.invoke(f"分析任务需要哪些技能：{task}")
    
    # 根据分析结果组合Skills
    return parse_skills(analysis)
```

---

### 创新点2：Web State Graph

**核心思想**：实时构建DOM快照有向图，让Agent能回溯、规划、纠错。

**新建文件**：
- `algorithms/state_graph.py`
- `utils/dom_snapshot.py`

**State Graph结构**：

```python
class WebStateGraph:
    """网页状态有向图"""
    
    def __init__(self):
        self.nodes = {}  # 节点：DOM快照ID → 快照数据
        self.edges = []  # 边：动作 → 状态变化
    
    def add_snapshot(self, dom_content, screenshot_path, action):
        """添加新状态节点"""
        node_id = generate_id()
        self.nodes[node_id] = {
            "dom": dom_content,
            "screenshot": screenshot_path,
            "action": action,
            "timestamp": time.now()
        }
        return node_id
    
    def backtrack(self, from_node_id, to_node_id):
        """回溯到之前的状态"""
        # 恢复之前的DOM和页面状态
        snapshot = self.nodes[to_node_id]
        return restore_state(snapshot)
    
    def find_path(self, start_id, goal_condition):
        """找到达目标的路径"""
        # 使用图搜索算法找路径
        return search_path(self.nodes, self.edges, goal_condition)
```

---

### Week 12：算法验证实验

**消融实验设计**：

| 版本 | 配置 | 预期成功率 |
|------|------|-----------|
| Baseline | 无创新算法 | 70% |
| +Adaptive Skill | 只加Skill Composition | 75% |
| +State Graph | 只加状态图 | 73% |
| +Both | 两者都加 | 80%+ |

---

---

## 第四阶段：大规模测试与优化（Week 13-16）

### 目标
用你的5万条数据全面测试，优化成功率，准备评估体系。

### 数据处理

#### Week 13：数据准备

**新建文件**：
- `data/loader.py`：数据加载器
- `data/schema.py`：数据格式定义
- `data/converter.py`：格式转换脚本

**数据格式定义（JSON Schema）**：

```json
{
    "task_id": "20260211-query-2586",
    "prompt": "打开豆瓣电影，找到评分最高的电影...",
    "expected_output": {
        "movie_name": "肖申克的救赎",
        "rating": "9.7"
    },
    "success_criteria": [
        "movie_name == '肖申克的救赎'",
        "rating >= 9.0"
    ],
    "start_url": "https://movie.douban.com/top250",
    "difficulty": "medium",
    "category": "data_extraction"
}
```

---

#### Week 14：批量测试

**新建文件**：
- `experiments/batch_runner.py`：批量执行脚本
- `experiments/result_analyzer.py`：结果分析

**批量执行流程**：

```python
async def run_batch(tasks: List[Task], agent: Agent):
    results = []
    for task in tasks:
        result = await agent.run(task.prompt)
        results.append({
            "task_id": task.task_id,
            "success": result.success,
            "steps": result.current_step,
            "time": result.end_time - result.start_time
        })
    return results
```

---

#### Week 15：评估体系

**新建文件**：
- `evaluation/judge.py`：LLM-as-Judge评估
- `evaluation/metrics.py`：评估指标计算

**LLM-as-Judge逻辑**：

```python
async def llm_judge(task_result, expected_output):
    """使用LLM判断任务是否成功"""
    prompt = f"""
    任务执行结果：{task_result.final_answer}
    预期输出：{expected_output}
    
    请判断执行结果是否正确匹配预期输出。
    输出格式：{"success": true/false, "reason": "原因"}
    """
    
    judgment = await llm.invoke(prompt)
    return parse_judgment(judgment)
```

---

---

## 第五阶段：产品化与论文（Week 17-20）

### Week 17：Web Demo

**新建文件**：
- `demo/app.py`：Gradio界面
- `demo/templates/`：HTML模板

**Demo功能**：
- 输入prompt → 实时看Agent日志 → 显示最终截图

---

### Week 18：FastAPI服务化

**新建文件**：
- `api/main.py`：FastAPI主服务
- `api/routes/task.py`：任务路由
- `docker/Dockerfile`：容器配置

---

### Week 19：开源准备

**文档内容**：
- README.md（已有，需完善）
- CONTRIBUTING.md：贡献指南
- ARCHITECTURE.md：架构说明

---

### Week 20：论文撰写

**论文结构**：

```
Title: WebClaw: MCP-Driven Deep Multi-Agent Framework for Large-Scale Real-World Web Tasks

1. Abstract（摘要）
2. Introduction（背景介绍）
3. Related Work（相关工作）
4. Method（方法）
   - 4.1 Multi-Agent Architecture
   - 4.2 Adaptive Skill Composition
   - 4.3 Web State Graph
5. Experiments（实验）
   - 5.1 WebTask50K Benchmark
   - 5.2 Ablation Study
6. Conclusion（结论）
```

---

---

## 总预算估算

| 项目 | 金额（人民币） |
|------|--------------|
| LLM调用费（DeepSeek为主） | 2000-3000 |
| 云GPU（可选，有本地GPU可跳过） | 500-1000 |
| 域名/存储/杂项 | 300-500 |
| **总计** | **3000-4500** |

---

## 你能从这个项目学到的技术

### 开发技术（偏工程）
| 技术 | 学习来源 | 掌握程度要求 |
|------|---------|-------------|
| LangGraph | agents/graph.py | 生产级掌握 |
| Playwright | tools/playwright_browser.py | 工业级自动化 |
| MCP | 后续阶段 | 协议标准 |
| FastAPI | api/main.py | 服务端开发 |
| Docker | docker/Dockerfile | 容器化部署 |
| Python Asyncio | 全项目异步代码 | 异步编程 |
| Pydantic | agents/state.py | 类型安全 |
| Redis | 后续阶段状态持久化 | 分布式存储 |

### 算法技术（偏研究）
| 技术 | 创新点 | 论文价值 |
|------|---------|---------|
| ReAct | agents/executor.py | 基础能力 |
| Multi-Agent | agents/graph.py | 高级编排 |
| DeepAgent | agents/deep_reasoner.py | 深度规划 |
| Adaptive Skill Composition | algorithms/skill_composer.py | **核心创新** |
| Web State Graph | algorithms/state_graph.py | **核心创新** |
| LLM-as-Judge | evaluation/judge.py | 标准评估 |

---

---

## 下一步行动

**Week 1 代码已全部生成，现在你需要做**：

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **配置API Key**：
   ```bash
   cp .env.example .env
   # 编辑.env，填入DEEPSEEK_API_KEY
   ```

3. **运行测试**：
   ```bash
   python tests/test_week1.py
   ```

4. **运行第一个任务**：
   ```bash
   python run.py --task "打开百度搜索Python"
   ```

**完成后回复我"Week 1 完成"，我给你Week 2的代码**。