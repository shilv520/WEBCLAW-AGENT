"""
Agent 状态定义 - WebClaw DeepAgent

本模块定义了 Agent 在 LangGraph 执行流程中传递的数据结构。

核心概念：
- TypedDict: Python 类型提示，用于定义特定键值类型的字典
- BaseModel: Pydantic 的基类，用于数据验证和序列化
- AgentState: 在 Agent 节点之间传递的主状态容器

作者: shilv
版本: 0.1.0
"""

from typing import TypedDict, List, Optional, Any
from pydantic import BaseModel, Field


class BrowserState(BaseModel):
    """
    浏览器状态容器 - 纯 Playwright 方式

    该类追踪 Playwright 浏览器实例的当前状态。
    截图直接保存为文件（PNG），路径存储在 screenshot_path 中。

    属性:
        website (str): 当前网站名称，如 "百度", "豆瓣", "GitHub"
        start_url (str): 起始 URL，如 "https://www.baidu.com"
        prompt (str): 用户的任务提示词
        url (str): 当前页面 URL（由 Playwright 实时获取）
        title (str): 当前页面标题
        completion_interface (str): 当前页面状态描述
        screenshot_path (str): 最新截图的文件路径（PNG格式）
    """
    website: str = ""
    start_url: str = ""
    prompt: str = ""
    url: str = ""
    title: str = ""
    completion_interface: str = ""
    screenshot_path: str = ""


class AgentState(TypedDict):
    """
    在 LangGraph 执行图中传递的主 Agent 状态

    这个 TypedDict 定义了在 LangGraph 流水线中
    不同 Agent 节点之间传递的所有状态变量。
    每个节点都可以读取和修改这些值。

    LangGraph 状态机制:
    - 状态在单个节点执行内是不可变的
    - 节点返回更新，这些更新会被合并到状态中
    - 状态在整个任务执行期间持久保存

    属性:
        task (str): 用户要执行的原始任务/提示词
        task_id (str): 用于追踪和日志记录的唯一标识符

        current_step (int): 当前执行步骤编号（从 0 开始）
                            用于防止无限循环
        max_steps (int): 最大允许步骤数，防止失控执行
                         默认为 20，平衡彻底性和成本
        step_history (List[dict]): 所有已执行步骤的完整历史
                                   每条记录包含: thought, action, result

        browser_state (BrowserState): 当前浏览器上下文
                                      包括 website、start_url、prompt、completion_interface

        thought (str): Agent 当前推理/思考过程
                       由 LLM 在每一步生成
        action (str): 要执行的具体动作（如 "click", "type"）
        action_result (str): 执行动作后的结果/观察

        final_answer (str): 任务成功完成时的最终输出

        execution_rate (float): 执行率（0.0-1.0）
                                计算公式: 成功步骤数 / 总步骤数
                                用于量化任务完成程度

        successful_steps (int): 成功执行的步骤数量
                                每个步骤执行成功则增加1

        success_threshold (float): 成功阈值（默认0.75）
                                当 execution_rate >= success_threshold 时，
                                任务判定为成功

        success (bool): 表示任务完成状态的布尔值
                       判断逻辑: success = (execution_rate >= success_threshold)
                       即执行率 >= 75% 则判定为成功

        error_message (str): 如果任务执行失败时的错误描述

        start_time (float): 任务执行开始时的 Unix 时间戳
        end_time (Optional[float]): 执行结束时的 Unix 时间戳
                                    任务仍在运行时为 None

    状态流转示例:
        初始状态 → Planner节点 → Executor节点 → Critic节点 → 最终状态

    注意:
        使用 TypedDict 而不是 Pydantic BaseModel，
        因为 LangGraph 期望可以部分更新的字典式状态。
    """
    # 输入部分 - Agent 接收的内容
    task: str                     # 用户任务/提示词描述
    task_id: str                   # 唯一标识符（如 "task_001"）

    # 执行追踪 - 进度监控
    current_step: int              # 当前步骤编号（0, 1, 2, ...）
    max_steps: int                 # 最大允许步骤数（安全限制）
    step_history: List[dict]       # 历史记录: {step, thought, action, result}

    # 浏览器上下文 - 当前网页状态
    browser_state: BrowserState    # Pydantic 模型，包含 URL、标题、DOM 等

    # 推理部分 - LLM 思考过程
    thought: str                   # Agent 当前步骤的推理
    action: str                    # 选择的动作名称（如 "click", "type"）
    action_result: str             # 动作执行后的观察/结果

    # 输出部分 - 最终结果
    final_answer: str              # 完成任务的输出（如果成功）
    execution_rate: float          # 执行率（成功步骤数/总步骤数，范围0.0-1.0）
    successful_steps: int          # 成功执行的步骤数量
    success_threshold: float       # 成功阈值（默认0.75，执行率>=阈值则成功）
    success: bool                  # success = (execution_rate >= success_threshold)
    error_message: str             # 错误详情（如果失败）

    # 元数据 - 时间信息
    start_time: float              # 执行开始时间戳
    end_time: Optional[float]      # 执行结束时间戳（运行中为 None）


def create_initial_state(task: str, task_id: str = "default") -> AgentState:
    """
    为新任务创建初始 AgentState 的工厂函数

    该函数初始化所有状态字段为默认值，
    为 Agent 执行流水线奠定基础。

    参数:
        task (str): 用户要执行的任务/提示词
                    示例: "打开百度搜索 Python 教程"
        task_id (str): 此任务实例的唯一标识符
                       用于日志和追踪。默认: "default"

    返回:
        AgentState: 准备好用于 LangGraph 执行的新状态字典
                    所有字段都初始化为合理的默认值

    状态初始化逻辑:
        - current_step = 0: 从零开始
        - max_steps = 20: 防止无限循环的安全限制
        - step_history = []: 空历史，执行时会填充
        - browser_state = BrowserState(): 空浏览器状态，未加载页面
        - execution_rate = 0.0: 初始执行率为0
        - successful_steps = 0: 初始成功步骤数为0
        - success_threshold = 0.75: 默认成功阈值75%
        - success = False: 初始标记为未完成（后续根据执行率计算）

    示例:
        >>> state = create_initial_state(
        ...     task="打开百度搜索 Python",
        ...     task_id="test_001"
        ... )
        >>> print(state["task"])
        "打开百度搜索 Python"
        >>> print(state["current_step"])
        0

    在 Agent 流水线中的使用:
        state = create_initial_state(task)
        result_state = await agent.run(state)
        print(result_state["success"])
    """
    import time  # 在此处导入避免顶层依赖

    return AgentState(
        # 输入初始化
        task=task,
        task_id=task_id,

        # 执行追踪 - 从步骤 0 开始
        current_step=0,
        max_steps=20,  # 大多数 Web 任务的合理限制
        step_history=[],  # 初始为空历史

        # 浏览器状态 - 空（尚未加载页面）
        browser_state=BrowserState(),

        # 推理 - 空字符串（将由 LLM 填充）
        thought="",
        action="",
        action_result="",

        # 输出 - 初始未完成
        final_answer="",
        execution_rate=0.0,  # 初始执行率为0
        successful_steps=0,  # 初始成功步骤数为0
        success_threshold=0.75,  # 默认成功阈值75%
        success=False,  # 任务完成时根据执行率计算
        error_message="",  # 执行失败时设置

        # 元数据 - 记录开始时间
        start_time=time.time(),  # 当前 Unix 时间戳
        end_time=None  # 尚未结束
    )


def calculate_execution_rate(state: AgentState) -> float:
    """
    计算任务执行率

    执行率 = 成功步骤数 / 总步骤数

    参数:
        state (AgentState): 当前 Agent 状态

    返回:
        float: 执行率（0.0-1.0）
    """
    total_steps = state["current_step"]
    if total_steps == 0:
        return 0.0
    return state["successful_steps"] / total_steps


def update_success_status(state: AgentState) -> AgentState:
    """
    根据执行率更新任务成功状态

    成功判断逻辑:
        success = (execution_rate >= success_threshold)
        即执行率 >= 75% 则判定为成功

    参数:
        state (AgentState): 当前 Agent 状态

    返回:
        AgentState: 更新后的状态（包含计算后的 execution_rate 和 success）

    示例:
        >>> state["successful_steps"] = 4  # 成功4步
        >>> state["current_step"] = 5      # 总共5步
        >>> state = update_success_status(state)
        >>> print(state["execution_rate"])
        0.8  # 4/5 = 80%
        >>> print(state["success"])
        True  # 80% >= 75%
    """
    # 计算执行率
    state["execution_rate"] = calculate_execution_rate(state)

    # 根据阈值判断是否成功
    state["success"] = state["execution_rate"] >= state["success_threshold"]

    return state