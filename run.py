"""
WebClaw DeepAgent - Main Entry Point

This script provides the command-line interface for running WebClaw tasks.
It handles:
- Argument parsing (task, model, headless mode)
- Configuration loading (API keys, model settings)
- Agent initialization and execution
- Result display and logging

Usage Examples:
    python run.py --task "打开百度搜索Python"
    python run.py --task "your task" --model deepseek-chat --headless
    python run.py --help

Author: WebClaw Team
Version: 0.1.0
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv
from loguru import logger

from agents.executor import ExecutorAgent


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

# Configure loguru for persistent logging
# Logs are saved to files with rotation and retention
logger.add(
    "logs/webclaw_{time}.log",  # Log file path pattern
    rotation="1 day",           # Create new file daily
    retention="7 days",         # Keep logs for 7 days
    level="INFO"                # Minimum log level
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

def get_model_config(model_name: str) -> dict:
    """
    Get API configuration for different LLM providers.

    Each provider has different:
    - API endpoint URL
    - Authentication method
    - Model naming convention

    This function centralizes configuration for easy switching.

    Args:
        model_name (str): Model identifier.
                          Supported: "deepseek-chat", "qwen-plus", "gpt-4o", "gpt-4o-mini"

    Returns:
        dict: Configuration containing:
            - api_key: Authentication key from environment
            - api_base: API endpoint URL
            - model: Actual model name to use

    Environment Variables Required:
        DEEPSEEK_API_KEY: For DeepSeek models
        QWEN_API_KEY: For Qwen models
        OPENAI_API_KEY: For OpenAI models

    Configuration Map:
        ┌─────────────────┐
        │  Model Request  │
        └─────────────────┘
               │
               ↓
        ┌─────────────────┐
        │ get_model_config│
        └─────────────────┘
               │
               ↓ returns
        ┌─────────────────┐
        │ Config Dict     │
        │ - api_key       │
        │ - api_base      │
        │ - model         │
        └─────────────────┘

    Example:
        >>> config = get_model_config("deepseek-chat")
        >>> print(config)
        {
            "api_key": "sk-xxx",
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
    """
    # Load environment variables from .env file
    load_dotenv()

    # Configuration map for different providers
    configs = {
        # DeepSeek (recommended - cost effective, good quality)
        "deepseek-chat": {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        },

        # Qwen (Alibaba Cloud - good for Chinese tasks)
        "qwen-plus": {
            "api_key": os.getenv("QWEN_API_KEY"),
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus"
        },

        # OpenAI GPT-4o (highest quality, higher cost)
        "gpt-4o": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": "https://api.openai.com/v1",
            "model": "gpt-4o"
        },

        # OpenAI GPT-4o-mini (faster, cheaper than GPT-4o)
        "gpt-4o-mini": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        },
    }

    # Return requested config (default to deepseek if not found)
    return configs.get(model_name, configs["deepseek-chat"])


# ============================================================================
# TASK EXECUTION
# ============================================================================

async def run_task(task: str, model: str = "deepseek-chat", headless: bool = False):
    """
    Execute a single web task using the ExecutorAgent.

    This function:
        1. Loads model configuration (API keys, endpoints)
        2. Creates ExecutorAgent instance
        3. Runs the task
        4. Displays results

    Args:
        task (str): User's task prompt description.
                    Example: "打开百度搜索Python教程"

        model (str): LLM model to use.
                     Default: "deepseek-chat"
                     Options: "deepseek-chat", "qwen-plus", "gpt-4o", "gpt-4o-mini"

        headless (bool): Browser visibility.
                         False: Show browser window (default)
                         True: Run invisibly (production)

    Returns:
        AgentState: Final execution state containing success status,
                    final answer, step history, etc.

    Execution Flow:
        ┌───────────────┐
        │  User Task    │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │ Load Config   │
        │ (API keys)    │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │ Create Agent  │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │ Execute Task  │
        │ (ReAct Loop)  │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │ Display Result│
        └───────────────┘

    Example:
        >>> result = await run_task("搜索Python", "deepseek-chat")
        >>> print(result["success"])
        True
    """
    # Log startup information
    logger.info(f"Starting WebClaw DeepAgent")
    logger.info(f"Task: {task}")
    logger.info(f"Model: {model}")

    # Get model configuration
    config = get_model_config(model)

    # Validate API key exists
    if not config["api_key"]:
        logger.error(f"API key not found for model: {model}")
        logger.error("Please set the corresponding environment variable in .env file")
        print("\n错误: API Key 未配置!")
        print(f"请在 .env 文件中设置 {model.upper().replace('-', '_')}_API_KEY")
        return None

    # Create agent instance
    agent = ExecutorAgent(
        model_name=config["model"],
        api_key=config["api_key"],
        api_base=config["api_base"],
        headless=headless
    )

    # Execute task
    result = await agent.run(task)

    # Display results to user
    print("\n" + "="*50)
    print("任务执行结果:")
    print("="*50)
    print(f"任务: {result['task']}")
    print(f"执行步骤数: {result['current_step']}")
    print(f"执行状态: {'成功' if result['success'] else '失败'}")

    if result['success']:
        print(f"\n最终答案: {result['final_answer']}")
    else:
        print(f"\n错误信息: {result['error_message']}")

    print("="*50)

    # Display step history for debugging
    print("\n执行步骤详情:")
    for step in result['step_history']:
        # Truncate result for readability
        result_preview = step['result'][:80] if len(step['result']) > 80 else step['result']
        print(f"  步骤{step['step']}: {step['action']} -> {result_preview}")

    return result


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Main entry point for command-line execution.

    This function:
        1. Parses command-line arguments
        2. Calls run_task() with parsed arguments
        3. Handles asyncio execution

    Command-Line Arguments:
        --task, -t: Required. The task prompt to execute.
        --model, -m: Optional. LLM model selection.
        --headless: Optional. Run browser invisibly.

    Argument Parsing with argparse:
        argparse provides:
        - Automatic help generation (--help)
        - Type validation
        - Default values
        - Required argument checking

    Asyncio Execution:
        asyncio.run() is the modern way to run async code.
        It creates event loop, runs coroutine, and closes loop.

    Example Commands:
        $ python run.py --task "打开百度搜索Python"
        $ python run.py -t "task" -m gpt-4o --headless
        $ python run.py --help
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="WebClaw DeepAgent - Web Task Execution Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --task "打开百度搜索Python"
  python run.py -t "打开豆瓣找评分最高的电影" -m deepseek-chat
  python run.py --task "task" --headless
        """
    )

    # Add arguments
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Task to execute (e.g., '打开百度搜索Python')"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-chat",
        choices=["deepseek-chat", "qwen-plus", "gpt-4o", "gpt-4o-mini"],
        help="LLM model to use (default: deepseek-chat)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (invisible)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run async task
    # asyncio.run() handles event loop creation and cleanup
    asyncio.run(run_task(args.task, args.model, args.headless))


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()