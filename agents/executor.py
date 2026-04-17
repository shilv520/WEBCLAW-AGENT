"""
Executor Agent - The Core Execution Engine for WebClaw DeepAgent

This module implements the primary agent that executes web tasks using
LLM reasoning + Playwright browser automation.

Week 1 Implementation: Single-agent ReAct-style executor.

Key Concepts:
- ReAct Pattern: Reasoning (Thought) → Acting (Action) → Observation (Result)
- LLM Decision Making: Using language models to decide next actions
- Playwright Integration: Real browser automation for web interactions

Architecture:
    User Task → ExecutorAgent.run() → [LLM Decision → Playwright Action] loop → Result

Author: WebClaw Team
Version: 0.1.0
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .state import AgentState, create_initial_state
from tools.playwright_browser import PlaywrightBrowser


# ============================================================================
# SYSTEM PROMPT - Instructions for the LLM Agent
# ============================================================================

EXECUTOR_SYSTEM_PROMPT = """
You are a Web Execution Agent that controls a browser to complete user tasks.

Available Tools:
- open_url(url): Open a specific website URL
- click(selector): Click an element on the page (CSS selector)
- type_text(selector, text): Type text into an input field
- scroll(direction): Scroll the page ("up" or "down")
- wait(seconds): Wait for a specified time
- extract_text(selector): Extract text content from an element
- screenshot(): Take a screenshot of the current page
- get_current_state(): Get current page URL, title, and content preview
- press_key(key): Press a keyboard key (e.g., "Enter", "Escape")
- find_elements(selector): Find all elements matching a CSS selector

Workflow:
1. Analyze the task and determine the first step
2. Choose the appropriate tool and parameters
3. Execute and observe the result
4. Decide the next step based on observation
5. Repeat until task is complete

Output Format (JSON):
{
    "thought": "Your reasoning process in Chinese",
    "action": "Tool name to execute",
    "action_args": {"param1": "value1", "param2": "value2"},
    "is_complete": false or true,
    "final_answer": "Final answer if complete, otherwise empty string"
}

IMPORTANT:
- Output ONLY the JSON object, no other text
- Use Chinese for the "thought" field
- Set is_complete=true only when the task is fully finished
"""


# ============================================================================
# EXECUTOR AGENT CLASS
# ============================================================================

class ExecutorAgent:
    """
    The executor agent that orchestrates LLM decisions and browser actions.

    This agent follows the ReAct pattern:
        1. Thought: LLM analyzes the current state and decides next action
        2. Action: Playwright executes the chosen browser operation
        3. Observation: Result is captured and added to context
        4. Loop: Repeat until task completion or max steps

    Architecture Diagram:
        ┌─────────────┐
        │   User Task │
        └─────────────┘
              ↓
        ┌─────────────┐     ┌─────────────┐
        │  LLM Decide │ ←── │   Context   │
        └─────────────┘     └─────────────┘
              ↓                    ↑
        ┌─────────────┐     ┌─────────────┐
        │  Playwright │ ──→ │ Observation │
        │   Execute   │     │   Result    │
        └─────────────┘     └─────────────┘
              ↓
        ┌─────────────┐
        │  Final Result│
        └─────────────┘

    Key Components:
        - llm: LangChain ChatOpenAI instance for decision making
        - browser: PlaywrightBrowser instance for web automation
        - state: AgentState tracking execution progress

    Attributes:
        llm (ChatOpenAI): The language model for reasoning decisions.
        browser (PlaywrightBrowser): Browser automation tool.

    Example:
        >>> agent = ExecutorAgent(
        ...     model_name="deepseek-chat",
        ...     api_key="your-api-key"
        ... )
        >>> result = await agent.run("Open Baidu and search Python")
        >>> print(result["success"])
        True
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        headless: bool = False
    ):
        """
        Initialize the executor agent with LLM and browser components.

        Args:
            model_name (str): The LLM model identifier.
                              Options: "deepseek-chat", "gpt-4o", "qwen-plus"
                              Default: "deepseek-chat" (cost-effective)

            api_key (Optional[str]): API key for the LLM service.
                                     Required for cloud-based models.
                                     Load from .env file if not provided.

            api_base (Optional[str]): Base URL for the API endpoint.
                                      Different providers have different URLs:
                                      - DeepSeek: https://api.deepseek.com/v1
                                      - OpenAI: https://api.openai.com/v1
                                      - Qwen: https://dashscope.aliyuncs.com/...

            headless (bool): Whether to run browser without visible window.
                             True: Background execution (production)
                             False: Visible browser (development/debugging)
                             Default: False for easier debugging

        Initialization Steps:
            1. Create LangChain ChatOpenAI instance with provided config
            2. Initialize Playwright browser with headless setting
            3. Log initialization status

        Example:
            >>> # Using DeepSeek (recommended for cost)
            >>> agent = ExecutorAgent(
            ...     model_name="deepseek-chat",
            ...     api_key="sk-xxx",
            ...     api_base="https://api.deepseek.com/v1"
            ... )

            >>> # Using OpenAI GPT-4
            >>> agent = ExecutorAgent(
            ...     model_name="gpt-4o",
            ...     api_key="sk-xxx"
            ... )
        """
        # Initialize LLM with LangChain ChatOpenAI wrapper
        # LangChain provides a unified interface for multiple LLM providers
        self.llm = ChatOpenAI(
            model=model_name,        # Model identifier
            api_key=api_key,         # Authentication key
            base_url=api_base,       # API endpoint URL
            temperature=0.7          # Creativity level (0=deterministic, 1=random)
        )

        # Initialize Playwright browser for web automation
        # Playwright provides reliable cross-browser automation
        self.browser = PlaywrightBrowser(headless=headless)

        # Log initialization status
        logger.info(f"ExecutorAgent initialized with model: {model_name}")

    # ------------------------------------------------------------------------
    # Main Execution Method
    # ------------------------------------------------------------------------

    async def run(self, task: str, task_id: str = "default") -> AgentState:
        """
        Execute a web task from start to completion.

        This is the main entry point for task execution. It orchestrates
        the entire ReAct loop: LLM decision → Browser action → Observation.

        Execution Flow:
            1. Create initial state from task
            2. Start Playwright browser
            3. Enter execution loop (max_steps iterations):
               a. Increment step counter
               b. Get LLM decision (which action to take)
               c. Execute action via Playwright
               d. Update state with thought, action, result
               e. Record step in history
               f. Check if task is complete
            4. Handle errors and edge cases
            5. Close browser and return final state

        Args:
            task (str): The user's task/prompt description.
                        Example: "Open Baidu and search for Python tutorials"

            task_id (str): Unique identifier for tracking/logging.
                           Default: "default"
                           Used in multi-task scenarios for distinction

        Returns:
            AgentState: Final state containing:
                        - success: True if task completed successfully
                        - final_answer: The task result
                        - step_history: Complete execution trace
                        - error_message: Error details if failed

        State Evolution Example:
            Initial: {task: "...", current_step: 0, success: False}
            Step 1:  {current_step: 1, thought: "...", action: "open_url"}
            Step 2:  {current_step: 2, thought: "...", action: "type_text"}
            Final:   {success: True, final_answer: "...", current_step: 3}

        Error Handling:
            - Catches all exceptions during execution
            - Sets error_message in state
            - Marks success as False
            - Ensures browser is always closed (finally block)

        Example:
            >>> result = await agent.run("Search Python on Baidu")
            >>> if result["success"]:
            ...     print(f"Answer: {result['final_answer']}")
            ... else:
            ...     print(f"Failed: {result['error_message']}")
        """
        # Step 1: Create initial state container
        # This sets up all tracking variables for execution
        state = create_initial_state(task, task_id)

        # Step 2: Start Playwright browser
        # This launches Chromium and creates a new page
        await self.browser.start()

        logger.info(f"Starting task: {task}")

        try:
            # Step 3: Main execution loop
            # Continue until max_steps reached or task completes
            while state["current_step"] < state["max_steps"]:

                # Increment step counter
                state["current_step"] += 1

                # Get LLM decision for next action
                # LLM analyzes current context and returns JSON decision
                decision = await self._get_llm_decision(state)

                # Execute the chosen action via Playwright
                # Returns result string (success message or error)
                action_result = await self._execute_action(
                    decision["action"],
                    decision["action_args"]
                )

                # Update state with current reasoning
                state["thought"] = decision["thought"]
                state["action"] = decision["action"]
                state["action_result"] = action_result

                # Record this step in history for later analysis
                # Each entry contains full step context
                state["step_history"].append({
                    "step": state["current_step"],
                    "thought": decision["thought"],
                    "action": decision["action"],
                    "action_args": decision["action_args"],
                    "result": action_result
                })

                # Log progress (truncated for readability)
                logger.info(
                    f"Step {state['current_step']}: {decision['action']} "
                    f"-> {action_result[:100]}"
                )

                # Check if task is complete
                # LLM sets is_complete=True when finished
                if decision.get("is_complete", False):
                    state["final_answer"] = decision.get("final_answer", "")
                    state["success"] = True
                    logger.success(f"Task completed: {state['final_answer']}")
                    break  # Exit execution loop

            # Handle max steps reached (timeout-like behavior)
            if state["current_step"] >= state["max_steps"]:
                state["error_message"] = "Max steps reached without completion"
                logger.warning("Max steps reached")

        except Exception as e:
            # Catch all exceptions during execution
            # Record error and mark as failed
            state["error_message"] = str(e)
            state["success"] = False
            logger.error(f"Task failed: {e}")

        finally:
            # Always close browser, even if exception occurred
            # This prevents resource leaks
            await self.browser.close()

            # Record end time for duration calculation
            import time
            state["end_time"] = time.time()

        return state

    # ------------------------------------------------------------------------
    # LLM Decision Making
    # ------------------------------------------------------------------------

    async def _get_llm_decision(self, state: AgentState) -> Dict[str, Any]:
        """
        Get the LLM's decision for the next action.

        This method:
            1. Builds context from current state (task, URL, history)
            2. Sends context to LLM with system prompt
            3. Parses LLM's JSON response into decision dict

        Context Building Logic:
            - Always include: task, current step, current URL
            - Include last 3 steps from history (for context continuity)
            - Include page title if available

        Args:
            state (AgentState): Current execution state containing
                                task, step history, and browser state

        Returns:
            Dict[str, Any]: LLM decision containing:
                - thought: Agent's reasoning (Chinese text)
                - action: Tool name (e.g., "click", "type_text")
                - action_args: Tool parameters (e.g., {"selector": "#search"})
                - is_complete: Boolean indicating task completion
                - final_answer: Final output (if complete)

        JSON Parsing Logic:
            - First attempt: Direct JSON parse of LLM response
            - Fallback: Regex extraction if response has extra text
            - Final fallback: Default "wait" action if parsing fails

        Example Response:
            {
                "thought": "需要先打开百度网站",
                "action": "open_url",
                "action_args": {"url": "https://www.baidu.com"},
                "is_complete": false,
                "final_answer": ""
            }
        """
        # Build context string from current state
        context = self._build_context(state)

        # Create message list for LLM
        # SystemMessage: Defines agent behavior and available tools
        # HumanMessage: Provides current context and task
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]

        # Invoke LLM asynchronously (non-blocking)
        response = await self.llm.ainvoke(messages)

        # Parse LLM response as JSON
        import json
        import re

        try:
            # Direct JSON parse (ideal case)
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: Extract JSON from response text
            # LLM sometimes adds extra text before/after JSON
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                # Final fallback: Default action if parsing completely fails
                decision = {
                    "thought": "Failed to parse LLM response",
                    "action": "wait",  # Safe default action
                    "action_args": {"seconds": 1},
                    "is_complete": False
                }

        return decision

    def _build_context(self, state: AgentState) -> str:
        """
        Build the context string for LLM decision making.

        The context provides the LLM with all necessary information
        to make an informed decision about the next action.

        Context Components:
            1. Task description (what to achieve)
            2. Current progress (step count)
            3. Current location (URL)
            4. Recent history (last 3 steps)
            5. Page info (title if available)

        Args:
            state (AgentState): Current execution state

        Returns:
            str: Formatted context string for LLM

        Example Output:
            任务: 打开百度搜索Python
            当前步骤: 2/20
            当前URL: https://www.baidu.com
            页面标题: 百度一下
            最近执行步骤:
            - 步骤1: open_url -> 已打开页面
            - 步骤2: type_text -> 已输入文字
        """
        context_parts = [
            f"任务: {state['task']}",
            f"当前步骤: {state['current_step']}/{state['max_steps']}",
            f"当前URL: {state['browser_state'].current_url or '未打开任何页面'}",
        ]

        # Add recent execution history (last 3 steps)
        # This gives LLM context continuity
        if state["step_history"]:
            recent_history = state["step_history"][-3:]
            history_text = "\n最近执行步骤:\n"
            for step in recent_history:
                # Truncate result for readability
                history_text += f"- 步骤{step['step']}: {step['action']} -> {step['result'][:50]}\n"
            context_parts.append(history_text)

        # Add page title if browser has loaded a page
        if state["browser_state"].page_title:
            context_parts.append(f"页面标题: {state['browser_state'].page_title}")

        return "\n".join(context_parts)

    # ------------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------------

    async def _execute_action(self, action: str, action_args: Dict[str, Any]) -> str:
        """
        Execute a browser action using Playwright.

        This method maps the LLM's action decision to actual Playwright
        browser operations. It handles both success and error cases.

        Action Mapping:
            Each action name maps to a PlaywrightBrowser method:
            - "open_url" → browser.open_url()
            - "click" → browser.click()
            - "type_text" → browser.type_text()
            - etc.

        Args:
            action (str): Action name from LLM decision
                          Must be a valid key in action_map

            action_args (Dict[str, Any]): Action parameters
                                           Passed as kwargs to browser method

        Returns:
            str: Result message from browser execution
                 Success: e.g., "已打开页面: https://..."
                 Error: e.g., "点击失败: element not found"

        Error Handling:
            - Unknown action: Returns error message
            - Browser error: Catches and returns error description
            - All errors are caught and returned as strings
              (not raised, to keep execution loop running)

        Example:
            >>> result = await self._execute_action("click", {"selector": "#search"})
            >>> print(result)
            "已点击元素: #search"
        """
        # Define action-to-method mapping
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
        }

        # Validate action name
        if action not in action_map:
            return f"Unknown action: {action}"

        # Execute action and handle errors
        try:
            # Call browser method with provided arguments
            result = await action_map[action](**action_args)
            return str(result)
        except Exception as e:
            # Return error message (don't raise exception)
            # This allows execution loop to continue
            return f"Action failed: {str(e)}"