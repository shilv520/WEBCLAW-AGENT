"""
轨迹记录器 - WebClaw DeepAgent

记录 Agent 执行 Web 任务的完整轨迹（state, action, reward, next_state），
为后续 RL 训练提供数据。

数据格式: JSONL，每行一条任务轨迹

作者: shilv
版本: 0.2.0
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class TrajectoryLogger:
    """
    轨迹记录器 - 记录 (state, action, reward, next_state) 四元组

    每个 Agent 执行步骤记录:
        - state: 执行前的状态（URL、DOM片段、步骤号）
        - action: 执行的动作（工具名、参数）
        - reward: 该步骤的奖励值
        - next_state: 执行后的状态

    轨迹完成后记录:
        - success: 任务是否成功
        - total_steps: 总步骤数
        - execution_rate: 执行率
        - duration: 总耗时
    """

    def __init__(self, output_dir: str = "./trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_trajectory: List[Dict[str, Any]] = []
        self._task_id: str = ""
        self._task: str = ""
        logger.info(f"TrajectoryLogger 已初始化，输出目录: {self.output_dir}")

    def start_task(self, task: str, task_id: str = "") -> None:
        """开始记录新任务轨迹"""
        if not task_id:
            task_id = f"task_{int(time.time())}"
        self._task_id = task_id
        self._task = task
        self._current_trajectory = []
        logger.info(f"开始记录轨迹: {task_id} - {task}")

    def log_step(
        self,
        step: int,
        state: Dict[str, Any],
        action: str,
        action_args: Dict[str, Any],
        action_result: str,
        thought: str,
        next_state: Dict[str, Any],
        reward: float = 0.0,
    ) -> None:
        """
        记录一个执行步骤

        参数:
            step: 步骤编号
            state: 执行前的状态快照
            action: 工具名称
            action_args: 工具参数
            action_result: 执行结果
            thought: LLM 思考过程
            next_state: 执行后的状态快照
            reward: 该步骤的奖励（默认0，后续由 RewardCalculator 计算）
        """
        self._current_trajectory.append({
            "step": step,
            "state": state,
            "action": {
                "tool": action,
                "args": action_args,
            },
            "thought": thought,
            "result": action_result,
            "next_state": next_state,
            "reward": reward,
            "timestamp": time.time(),
        })

    def end_task(
        self,
        success: bool,
        final_answer: str = "",
        error_message: str = "",
        execution_rate: float = 0.0,
        total_steps: int = 0,
    ) -> str:
        """
        结束任务并保存轨迹到 JSONL 文件

        返回:
            str: 保存的文件路径
        """
        duration = 0.0
        if self._current_trajectory:
            duration = (
                self._current_trajectory[-1]["timestamp"]
                - self._current_trajectory[0]["timestamp"]
            )

        trajectory_data = {
            "task_id": self._task_id,
            "task": self._task,
            "success": success,
            "final_answer": final_answer,
            "error_message": error_message,
            "execution_rate": execution_rate,
            "total_steps": total_steps,
            "duration_seconds": round(duration, 2),
            "trajectory": self._current_trajectory,
        }

        filename = f"{self._task_id}.jsonl"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"轨迹已保存: {filepath} "
            f"(成功={success}, 步数={total_steps}, 耗时={duration:.1f}s)"
        )

        # 重置
        self._current_trajectory = []
        self._task_id = ""
        self._task = ""

        return str(filepath)

    def get_current_trajectory(self) -> List[Dict[str, Any]]:
        """返回当前正在记录的轨迹"""
        return self._current_trajectory


def extract_state_snapshot(browser_state) -> Dict[str, Any]:
    """
    从 BrowserState 提取状态快照（用于轨迹记录）

    参数:
        browser_state: BrowserState Pydantic 模型

    返回:
        Dict: 状态快照
    """
    return {
        "website": browser_state.website,
        "url": browser_state.url,
        "title": browser_state.title,
        "start_url": browser_state.start_url,
        "completion_interface": browser_state.completion_interface,
    }