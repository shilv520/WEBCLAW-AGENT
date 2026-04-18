"""
执行器代理 - WebClaw DeepAgent 的核心执行引擎

本模块实现了使用 LLM 推理 + Playwright 浏览器自动化
来执行 Web 任务的主要代理。

阶段1 实现：单代理 ReAct 风格执行器。

核心概念:
- ReAct 模式: Reasoning（思考）→ Acting（行动）→ Observation（观察）
- LLM 决策: 使用语言模型决定下一步动作
- Playwright 集成: 真实浏览器自动化进行 Web 交互

架构流程:
    用户任务 → ExecutorAgent.run() → [LLM决策 → Playwright动作] 循环 → 结果

作者: WebClaw Team
版本: 0.1.0
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .state import AgentState, create_initial_state, update_success_status
from .trajectory_logger import TrajectoryLogger, extract_state_snapshot
from tools.playwright_browser import PlaywrightBrowser


# ============================================================================
# 系统提示词 - LLM Agent 的指令
# ============================================================================

EXECUTOR_SYSTEM_PROMPT = """
你是一个 Web 执行代理，你的任务是控制浏览器完成用户的任务。

你有以下工具可以使用：
- open_url(url): 打开指定的网页 URL
- click(selector): 点击页面上的元素（CSS 选择器）
- type_text(selector, text): 在输入框中输入文字
- scroll(direction): 滚动页面（"up" 或 "down"）
- wait(seconds): 等待指定秒数
- extract_text(selector): 提取元素的文本内容
- screenshot(): 截取当前页面截图
- get_current_state(): 获取当前页面 URL、标题和内容预览
- press_key(key): 按下键盘按键（如 "Enter", "Escape"）
- find_elements(selector): 查找所有匹配 CSS 选择器的元素
- get_interactive_elements(): 获取页面上所有可交互元素（输入框、按钮、链接等）

工作流程：
1. 打开页面后，先用 get_interactive_elements() 了解页面有哪些元素
2. 根据元素信息选择正确的 selector 进行操作
3. 执行操作后观察结果
4. 如果失败，尝试其他 selector 或等待页面加载
5. 重复直到任务完成

输出格式（JSON）：
{
    "thought": "你的思考过程（用中文）",
    "action": "要执行的工具名称",
    "action_args": {"参数名": "参数值"},
    "is_complete": false 或 true,
    "final_answer": "如果完成，填写最终答案，否则为空字符串"
}

重要提示：
- 只输出 JSON 对象，不要输出其他内容
- thought 字段使用中文
- 打开新页面后，先调用 get_interactive_elements() 了解页面结构
- 不要猜测 selector，先用 get_interactive_elements() 找到正确的 selector
- 只有任务完全完成时才设置 is_complete=true
"""


# ============================================================================
# 执行器代理类
# ============================================================================

class ExecutorAgent:
    """
    编排 LLM 决策和浏览器动作的执行器代理

    该代理遵循 ReAct 模式：
        1. Thought（思考）: LLM 分析当前状态并决定下一步动作
        2. Action（行动）: Playwright 执行选择的浏览器操作
        3. Observation（观察）: 捕获结果并添加到上下文
        4. Loop（循环）: 重复直到任务完成或达到最大步骤

    架构流程图:
        ┌─────────────┐
        │   用户任务   │
        └─────────────┘
              ↓
        ┌─────────────┐     ┌─────────────┐
        │  LLM 决策   │ ←── │   上下文    │
        └─────────────┘     └─────────────┘
              ↓                    ↑
        ┌─────────────┐     ┌─────────────┐
        │  Playwright │ ──→ │   观察结果  │
        │   执行动作  │     │            │
        └─────────────┘     └─────────────┘
              ↓
        ┌─────────────┐
        │  最终结果   │
        └─────────────┘

    核心组件:
        - llm: LangChain ChatOpenAI 实例，用于决策
        - browser: PlaywrightBrowser 实例，用于 Web 自动化
        - state: AgentState 追踪执行进度

    属性:
        llm (ChatOpenAI): 用于推理决策的语言模型
        browser (PlaywrightBrowser): 浏览器自动化工具

    示例:
        >>> agent = ExecutorAgent(
        ...     model_name="deepseek-chat",
        ...     api_key="your-api-key"
        ... )
        >>> result = await agent.run("打开百度搜索 Python")
        >>> print(result["success"])
        True
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        headless: bool = False,
        enable_trajectory: bool = True,
    ):
        """
        使用 LLM 和浏览器组件初始化执行器代理

        参数:
            model_name (str): LLM 模型标识符
                              选项: "deepseek-chat", "gpt-4o", "qwen-plus"
                              默认: "deepseek-chat"（性价比高）

            api_key (Optional[str]): LLM 服务 API 密钥
                                     云端模型必需
                                     如未提供，从 .env 文件加载

            api_base (Optional[str]): API 端点的基础 URL
                                      不同提供商有不同 URL:
                                      - DeepSeek: https://api.deepseek.com/v1
                                      - OpenAI: https://api.openai.com/v1
                                      - Qwen: https://dashscope.aliyuncs.com/...

            headless (bool): 是否在无可见窗口模式下运行浏览器
                             True: 后台执行（生产环境）
                             False: 可见浏览器（开发/调试）
                             默认: False 以便于调试

        初始化步骤:
            1. 使用提供的配置创建 LangChain ChatOpenAI 实例
            2. 使用 headless 设置初始化 Playwright 浏览器
            3. 记录初始化状态日志

        示例:
            >>> # 使用 DeepSeek（推荐，性价比高）
            >>> agent = ExecutorAgent(
            ...     model_name="deepseek-chat",
            ...     api_key="sk-xxx",
            ...     api_base="https://api.deepseek.com/v1"
            ... )

            >>> # 使用 OpenAI GPT-4
            >>> agent = ExecutorAgent(
            ...     model_name="gpt-4o",
            ...     api_key="sk-xxx"
            ... )
        """
        # 使用 LangChain ChatOpenAI 包装器初始化 LLM
        # LangChain 为多个 LLM 提供商提供统一接口
        self.llm = ChatOpenAI(
            model=model_name,        # 模型标识符
            api_key=api_key,         # 认证密钥
            base_url=api_base,       # API 端点 URL
            temperature=0.7          # 创造性程度（0=确定性，1=随机）
        )

        # 初始化 Playwright 浏览器用于 Web 自动化
        # Playwright 提供可靠的跨浏览器自动化
        self.browser = PlaywrightBrowser(headless=headless)

        # 轨迹记录器（阶段2新增，供 RL 训练用）
        self.enable_trajectory = enable_trajectory
        self.trajectory_logger = TrajectoryLogger() if enable_trajectory else None

        logger.info(f"ExecutorAgent 已初始化，模型: {model_name}, 轨迹记录: {enable_trajectory}")

    # ------------------------------------------------------------------------
    # 主执行方法
    # ------------------------------------------------------------------------

    async def run(self, task: str, task_id: str = "default") -> AgentState:
        """
        从开始到完成执行一个 Web 任务

        这是任务执行的主要入口点。它编排整个 ReAct 循环：
        LLM 决策 → 浏览器动作 → 观察。

        执行流程:
            1. 从任务创建初始状态
            2. 启动 Playwright 浏览器
            3. 进入执行循环（最多 max_steps 次迭代）:
               a. 增加步骤计数器
               b. 获取 LLM 决策（选择哪个动作）
               c. 通过 Playwright 执行动作
               d. 用 thought、action、result 更新状态
               e. 在历史中记录此步骤
               f. 检查任务是否完成
            4. 处理错误和边界情况
            5. 关闭浏览器并返回最终状态

        参数:
            task (str): 用户的任务/提示词描述
                        示例: "打开百度搜索 Python 教程"

            task_id (str): 用于追踪/日志的唯一标识符
                           默认: "default"
                           在多任务场景中用于区分

        返回:
            AgentState: 包含以下内容的最终状态:
                        - success: 任务是否成功完成
                        - final_answer: 任务结果
                        - step_history: 完整执行轨迹
                        - error_message: 如果失败的错误详情

        状态演变示例:
            初始: {task: "...", current_step: 0, success: False}
            步骤1: {current_step: 1, thought: "...", action: "open_url"}
            步骤2: {current_step: 2, thought: "...", action: "type_text"}
            最终: {success: True, final_answer: "...", current_step: 3}

        错误处理:
            - 捕获执行过程中的所有异常
            - 在状态中设置 error_message
            - 将 success 标记为 False
            - 确保浏览器总是关闭（finally 块）

        示例:
            >>> result = await agent.run("在百度搜索 Python")
            >>> if result["success"]:
            ...     print(f"答案: {result['final_answer']}")
            ... else:
            ...     print(f"失败: {result['error_message']}")
        """
        # 步骤1: 创建初始状态容器
        state = create_initial_state(task, task_id)

        # 设置浏览器状态的 prompt 字段
        state["browser_state"].prompt = task

        # 开始轨迹记录（阶段2新增）
        if self.trajectory_logger:
            self.trajectory_logger.start_task(task, task_id)

        # 步骤2: 启动 Playwright 浏览器
        await self.browser.start()

        logger.info(f"开始执行任务: {task}")

        try:
            # 步骤3: 主执行循环
            while state["current_step"] < state["max_steps"]:

                # 记录执行前状态（用于轨迹）
                pre_state_snapshot = extract_state_snapshot(state["browser_state"])

                # 增加步骤计数器
                state["current_step"] += 1

                # 获取 LLM 对下一步动作的决策
                decision = await self._get_llm_decision(state)

                # 通过 Playwright 执行选择的动作
                action_result = await self._execute_action(
                    decision["action"],
                    decision["action_args"]
                )

                # 记录执行后状态（用于轨迹）
                post_state_snapshot = extract_state_snapshot(state["browser_state"])

                # 用当前推理更新状态
                state["thought"] = decision["thought"]
                state["action"] = decision["action"]
                state["action_result"] = action_result

                # 记录轨迹步骤（阶段2新增）
                if self.trajectory_logger:
                    self.trajectory_logger.log_step(
                        step=state["current_step"],
                        state=pre_state_snapshot,
                        action=decision["action"],
                        action_args=decision["action_args"],
                        action_result=action_result,
                        thought=decision["thought"],
                        next_state=post_state_snapshot,
                    )

                # 在历史中记录此步骤
                state["step_history"].append({
                    "step": state["current_step"],
                    "thought": decision["thought"],
                    "action": decision["action"],
                    "action_args": decision["action_args"],
                    "result": action_result
                })

                logger.info(
                    f"步骤 {state['current_step']}: {decision['action']} "
                    f"-> {action_result[:100]}"
                )

                if decision.get("is_complete", False):
                    state["final_answer"] = decision.get("final_answer", "")
                    state["success"] = True
                    logger.success(f"任务完成: {state['final_answer']}")
                    break

            if state["current_step"] >= state["max_steps"]:
                state["error_message"] = "达到最大步骤数仍未完成"
                logger.warning("达到最大步骤数")

        except Exception as e:
            state["error_message"] = str(e)
            state["success"] = False
            logger.error(f"任务失败: {e}")

        finally:
            await self.browser.close()

            import time
            state["end_time"] = time.time()

            # 结束轨迹记录并保存（阶段2新增）
            if self.trajectory_logger:
                state = update_success_status(state)
                self.trajectory_logger.end_task(
                    success=state["success"],
                    final_answer=state["final_answer"],
                    error_message=state["error_message"],
                    execution_rate=state["execution_rate"],
                    total_steps=state["current_step"],
                )

        return state

    # ------------------------------------------------------------------------
    # LLM 决策方法
    # ------------------------------------------------------------------------

    async def _get_llm_decision(self, state: AgentState) -> Dict[str, Any]:
        """
        获取 LLM 对下一步动作的决策

        该方法:
            1. 从当前状态构建上下文（任务、URL、历史）
            2. 将上下文和系统提示词发送给 LLM
            3. 将 LLM 的 JSON 响应解析为决策字典

        上下文构建逻辑:
            - 总是包含: 任务、当前步骤、当前 URL
            - 包含历史的最近 3 个步骤（保持上下文连续性）
            - 如果可用，包含页面标题

        参数:
            state (AgentState): 当前执行状态，包含
                                任务、步骤历史、浏览器状态

        返回:
            Dict[str, Any]: LLM 决策，包含:
                - thought: Agent 推理（中文文本）
                - action: 工具名称（如 "click", "type_text"）
                - action_args: 工具参数（如 {"selector": "#search"}）
                - is_complete: 表示任务完成的布尔值
                - final_answer: 最终输出（如果完成）

        JSON 解析逻辑:
            - 首先尝试: 直接解析 LLM 响应的 JSON
            - 后备方案: 如果响应有额外文本，用正则提取
            - 最终后备: 如果解析完全失败，默认 "wait" 动作

        响应示例:
            {
                "thought": "需要先打开百度网站",
                "action": "open_url",
                "action_args": {"url": "https://www.baidu.com"},
                "is_complete": false,
                "final_answer": ""
            }
        """
        # 从当前状态构建上下文字符串
        context = self._build_context(state)

        # 为 LLM 创建消息列表
        # SystemMessage: 定义 Agent 行为和可用工具
        # HumanMessage: 提供当前上下文和任务
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]

        # 异步调用 LLM（非阻塞）
        response = await self.llm.ainvoke(messages)

        # 将 LLM 响应解析为 JSON
        import json
        import re

        try:
            # 直接 JSON 解析（理想情况）
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            # 后备方案: 从响应文本中提取 JSON
            # LLM 有时会在 JSON 前后添加额外文本
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                # 最终后备: 如果解析完全失败，使用默认动作
                decision = {
                    "thought": "无法解析 LLM 响应",
                    "action": "wait",  # 安全的默认动作
                    "action_args": {"seconds": 1},
                    "is_complete": False
                }

        return decision

    def _build_context(self, state: AgentState) -> str:
        """
        为 LLM 决策构建上下文字符串

        上下文为 LLM 提供做出明智决策所需的所有必要信息。

        上下文组件:
            1. 任务提示词（prompt）
            2. 当前网站（website）
            3. 起始URL（start_url）
            4. 当前进度（步骤计数）
            5. 任务完成界面描述（completion_interface）
            6. 最近历史（最近 3 个步骤）

        参数:
            state (AgentState): 当前执行状态

        返回:
            str: 为 LLM 格式化的上下文字符串

        输出示例:
            任务: 打开百度搜索Python
            网站: 百度
            起始URL: https://www.baidu.com
            当前步骤: 2/20
            完成界面: 搜索结果页面
            最近执行步骤:
            - 步骤1: open_url -> 已打开页面
            - 步骤2: type_text -> 已输入文字
        """
        context_parts = [
            f"任务: {state['task']}",
            f"网站: {state['browser_state'].website or '未打开任何网站'}",
            f"当前URL: {state['browser_state'].url or '未设置'}",
            f"页面标题: {state['browser_state'].title or '未知'}",
            f"当前步骤: {state['current_step']}/{state['max_steps']}",
        ]

        # 添加最近的执行历史（最近 3 个步骤）
        # 这给 LLM 提供上下文连续性
        if state["step_history"]:
            recent_history = state["step_history"][-3:]
            history_text = "\n最近执行步骤:\n"
            for step in recent_history:
                # 截断结果以便阅读
                history_text += f"- 步骤{step['step']}: {step['action']} -> {step['result'][:50]}\n"
            context_parts.append(history_text)

        # 如果浏览器已加载页面，添加完成界面描述
        if state["browser_state"].completion_interface:
            context_parts.append(f"当前页面状态: {state['browser_state'].completion_interface}")

        return "\n".join(context_parts)

    # ------------------------------------------------------------------------
    # 动作执行方法
    # ------------------------------------------------------------------------

    async def _execute_action(self, action: str, action_args: Dict[str, Any]) -> str:
        """
        使用 Playwright 执行浏览器动作

        该方法将 LLM 的动作决策映射到实际的 Playwright
        浏览器操作。它处理成功和错误两种情况。

        动作映射:
            每个动作名称映射到一个 PlaywrightBrowser 方法:
            - "open_url" → browser.open_url()
            - "click" → browser.click()
            - "type_text" → browser.type_text()
            - 等等

        参数:
            action (str): LLM 决策中的动作名称
                          必须是 action_map 中的有效键

            action_args (Dict[str, Any]): 动作参数
                                           作为 kwargs 传递给浏览器方法

        返回:
            str: 浏览器执行的结果消息
                 成功: 如 "已打开页面: https://..."
                 错误: 如 "点击失败: element not found"

        错误处理:
            - 未知动作: 返回错误消息
            - 浏览器错误: 捕获并返回错误描述
            - 所有错误都被捕获并作为字符串返回
              （不抛出异常，保持执行循环运行）

        示例:
            >>> result = await self._execute_action("click", {"selector": "#search"})
            >>> print(result)
            "已点击元素: #search"
        """
        # 定义动作到方法的映射
        action_map = {
            "open_url": self.browser.open_url,
            "click": self.browser.click,
            "type_text": self.browser.type_text,
            "scroll": self.browser.scroll,
            "wait": self.browser.wait,
            "extract_text": self.browser.extract_text,
            "screenshot": self.browser.screenshot,
            "get_current_state": self.browser.get_current_state,
            "press_key": self.browser.press_key,
            "find_elements": self.browser.find_elements,
            "get_interactive_elements": self.browser.get_interactive_elements,
        }

        # 验证动作名称
        if action not in action_map:
            return f"未知动作: {action}"

        # 执行动作并处理错误
        try:
            # 用提供的参数调用浏览器方法
            result = await action_map[action](**action_args)
            return str(result)
        except Exception as e:
            # 返回错误消息（不抛出异常）
            # 这允许执行循环继续
            return f"动作执行失败: {str(e)}"