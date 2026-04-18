"""
WebClaw DeepAgent - 主入口点

该脚本提供运行 WebClaw 任务的命令行接口。
它处理:
- 参数解析（任务、模型、无头模式）
- 配置加载（API 密钥、模型设置）
- 代理初始化和执行
- 结果显示和日志

使用示例:
    python run.py --task "打开百度搜索Python"
    python run.py --task "你的任务" --model deepseek-chat --headless
    python run.py --help

作者: xdshilv
版本: 0.1.0
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv
from loguru import logger

from agents.executor import ExecutorAgent


# ============================================================================
# 日志配置
# ============================================================================

# 配置 loguru 用于持久化日志
# 日志保存到文件，有轮转和保留策略
logger.add(
    "logs/webclaw_{time}.log",  # 日志文件路径模式
    rotation="1 day",           # 每天创建新文件
    retention="7 days",         # 保留日志7天
    level="INFO"                # 最小日志级别
)


# ============================================================================
# 模型配置
# ============================================================================

def get_model_config(model_name: str) -> dict:
    """
    获取不同 LLM 提供商的 API 配置

    每个提供商有不同的:
    - API 端点 URL
    - 认证方式
    - 模型命名约定

    该函数集中配置，便于切换。

    参数:
        model_name (str): 模型标识符
                          支持: "deepseek-chat", "qwen-plus", "gpt-4o", "gpt-4o-mini"

    返回:
        dict: 配置字典，包含:
            - api_key: 来自环境变量的认证密钥
            - api_base: API 端点 URL
            - model: 实际使用的模型名称

    需要的环境变量:
        DEEPSEEK_API_KEY: 用于 DeepSeek 模型
        QWEN_API_KEY: 用于 Qwen 模型
        OPENAI_API_KEY: 用于 OpenAI 模型

    配置映射:
        ┌─────────────────┐
        │   模型请求      │
        └─────────────────┘
               │
               ↓
        ┌─────────────────┐
        │ get_model_config│
        └─────────────────┘
               │
               ↓ 返回
        ┌─────────────────┐
        │  配置字典       │
        │ - api_key       │
        │ - api_base      │
        │ - model         │
        └─────────────────┘

    示例:
        >>> config = get_model_config("deepseek-chat")
        >>> print(config)
        {
            "api_key": "sk-xxx",
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
    """
    # 从 .env 文件加载环境变量
    load_dotenv()

    # 不同提供商的配置映射
    configs = {
        # OpenAI GPT-4o（最高质量，成本较高）
        "gpt-4o": {
            "api_key": "hk-9wcj8410000549565f962b4c7c68595e71e789d4e40ca287",
            "api_base": "https://api.openai-hk.com/v1",
            "model": "gpt-4o"
        },

        "qwen-plus": {
            "api_key": os.getenv("QWEN_API_KEY"),
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus"
        },

        # OpenAI GPT-4o-mini（比 GPT-4o 更快更便宜）
        "gpt-4o-mini": {
            "api_key": "hk-9wcj8410000549565f962b4c7c68595e71e789d4e40ca287",
            "api_base": "https://api.openai-hk.com/v1",
            "model": "gpt-4o-mini"
        },
    }

    # 返回请求的配置（未找到则默认使用 gpt-4o-mini）
    return configs.get(model_name, configs["gpt-4o-mini"])


# ============================================================================
# 任务执行
# ============================================================================

async def run_task(task: str, model: str = "gpt-4o", headless: bool = False):
    """
    使用 ExecutorAgent 执行单个 Web 任务

    该函数:
        1. 加载模型配置（API 密钥、端点）
        2. 创建 ExecutorAgent 实例
        3. 运行任务
        4. 显示结果

    参数:
        task (str): 用户任务提示词描述
                    示例: "打开百度搜索Python教程"

        model (str): 使用的 LLM 模型
                     默认: "deepseek-chat"
                     选项: "deepseek-chat", "qwen-plus", "gpt-4o", "gpt-4o-mini"

        headless (bool): 浏览器可见性
                         False: 显示浏览器窗口（默认）
                         True: 不可见运行（生产环境）

    返回:
        AgentState: 最终执行状态，包含成功状态、
                    最终答案、步骤历史等

    执行流程:
        ┌───────────────┐
        │   用户任务    │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │  加载配置    │
        │ (API 密钥)   │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │  创建代理    │
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │  执行任务    │
        │ (ReAct 循环)│
        └───────────────┘
               │
               ↓
        ┌───────────────┐
        │  显示结果    │
        └───────────────┘

    示例:
        >>> result = await run_task("搜索Python", "deepseek-chat")
        >>> print(result["success"])
        True
    """
    # 记录启动信息
    logger.info(f"正在启动 WebClaw DeepAgent")
    logger.info(f"任务: {task}")
    logger.info(f"模型: {model}")

    # 获取模型配置
    config = get_model_config(model)

    # 验证 API 密钥存在
    if not config["api_key"]:
        logger.error(f"未找到模型 {model} 的 API 密钥")
        logger.error("请在 .env 文件中设置对应的环境变量")
        print("\n错误: API Key 未配置!")
        print(f"请在 .env 文件中设置 {model.upper().replace('-', '_')}_API_KEY")
        return None

    # 创建代理实例
    agent = ExecutorAgent(
        model_name=config["model"],
        api_key=config["api_key"],
        api_base=config["api_base"],
        headless=headless
    )

    # 执行任务
    result = await agent.run(task)

    # 向用户显示结果
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

    # 显示步骤历史用于调试
    print("\n执行步骤详情:")
    for step in result['step_history']:
        # 截断结果以便阅读
        result_preview = step['result'][:80] if len(step['result']) > 80 else step['result']
        print(f"  步骤{step['step']}: {step['action']} -> {result_preview}")

    return result


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """
    命令行执行的主入口点

    该函数:
        1. 解析命令行参数
        2. 用解析的参数调用 run_task()
        3. 处理 asyncio 执行

    命令行参数:
        --task, -t: 必需。要执行的任务提示词。
        --model, -m: 可选。LLM 模型选择。
        --headless: 可选。不可见运行浏览器。

    使用 argparse 解析参数:
        argparse 提供:
        - 自动生成帮助（--help）
        - 类型验证
        - 默认值
        - 必需参数检查

    Asyncio 执行:
        asyncio.run() 是运行异步代码的现代方式。
        它创建事件循环、运行协程、关闭循环。

    命令示例:
        $ python run.py --task "打开百度搜索Python"
        $ python run.py -t "任务" -m gpt-4o --headless
        $ python run.py --help
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="WebClaw DeepAgent - Web 任务执行代理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --task "打开百度搜索Python"
  python run.py -t "打开豆瓣找评分最高的电影" -m deepseek-chat
  python run.py --task "任务" --headless
        """
    )

    # 添加参数
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="要执行的任务（如 '打开百度搜索Python'）"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        choices=["qwen-plus", "gpt-4o", "gpt-4o-mini"],
        help="使用的 LLM 模型（默认: gpt-4o-mini）"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="在无头模式下运行浏览器（不可见）"
    )

    # 解析参数
    args = parser.parse_args()

    # 运行异步任务
    # asyncio.run() 处理事件循环的创建和清理
    asyncio.run(run_task(args.task, args.model, args.headless))


# ============================================================================
# 脚本入口点
# ============================================================================

if __name__ == "__main__":
    main()