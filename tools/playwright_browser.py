"""
Playwright 浏览器工具 - WebClaw DeepAgent 的浏览器自动化核心引擎

本模块提供了基于 Playwright 的全面浏览器自动化工具，
使代理能够像人类用户一样与网页交互。

核心功能:
- 页面导航: 打开 URL、浏览页面
- 元素交互: 点击、输入、滚动、按键
- 内容提取: 获取文本、HTML、元素信息
- 视觉捕获: 截图用于调试和验证
- 状态追踪: 维护当前浏览器状态

架构:
    ┌─────────────────┐
    │  ExecutorAgent  │
    └─────────────────┘
           │ 调用
           ↓
    ┌─────────────────┐
    │ PlaywrightBrowser│
    │  ┌───────────┐  │
    │  │ Playwright│  │ ← 浏览器自动化库
    │  │  Chromium │  │ ← 浏览器实例
    │  │   Page    │  │ ← 网页上下文
    │  └───────────┘  │
    └─────────────────┘
           │
           ↓
    ┌─────────────────┐
    │   Web Page      │
    │ (目标网站)      │
    └─────────────────┘

异步编程说明:
    所有方法都是 async（使用 async/await），因为:
    - 浏览器操作是 I/O 密集型（网络、渲染）
    - async 允许非阻塞执行
    - 多个操作可以并发运行

作者: xdshilv
版本: 0.1.0
"""

import asyncio
import sys
from pathlib import Path

# 确保 agents 目录在路径中，以便导入 BrowserState
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any, List
from loguru import logger

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from agents.state import BrowserState


# ============================================================================
# Playwright 浏览器类
# ============================================================================

class PlaywrightBrowser:
    """
    基于 Playwright 的浏览器自动化工具，用于 Web 代理执行

    该类封装了 Playwright 的异步 API，提供简洁、代理友好的
    浏览器操作接口。它处理:
    - 浏览器生命周期（启动/关闭）
    - 页面导航和交互
    - 内容提取
    - 视觉捕获（截图）
    - 状态追踪

    使用模式:
        1. 用配置初始化（headless、timeout 等）
        2. 调用 start() 启动浏览器
        3. 执行操作（open_url、click、type 等）
        4. 调用 close() 清理资源

    关键设计决策:
        - 异步方法: 非阻塞 I/O 提高性能
        - 自动截图: 捕获状态用于调试
        - 错误处理: 返回错误消息而不是抛出异常
        - 状态追踪: 维护 BrowserState 用于上下文

    属性:
        headless (bool): 无可见浏览器窗口运行
        timeout (int): 默认操作超时时间（毫秒）
        screenshot_dir (Path): 保存截图的目录

        playwright: Playwright 实例（来自 async_playwright）
        browser: Chromium 浏览器实例
        context: 浏览器上下文（隔离会话）
        page: 当前网页

        current_state: 追踪的浏览器状态
        step_count: 用于命名截图的计数器

    示例:
        >>> browser = PlaywrightBrowser(headless=False)
        >>> await browser.start()
        >>> await browser.open_url("https://www.baidu.com")
        >>> await browser.type_text("#kw", "Python")
        >>> await browser.click("#su")
        >>> await browser.close()
    """

    def __init__(
        self,
        headless: bool = False,
        timeout: int = 30000,
        screenshot_dir: str = "./screenshots"
    ):
        """
        用配置初始化浏览器工具

        参数:
            headless (bool): 浏览器可见性模式
                             False: 显示浏览器窗口（用于调试）
                             True: 不可见运行（用于生产）
                             默认: False（开发时推荐）

            timeout (int): 操作的默认超时时间（毫秒）
                           网络操作在此时间后失败
                           默认: 30000（30秒）
                           对于慢网络/网站需增加

            screenshot_dir (str): 保存截图的目录路径
                                  截图用于:
                                  - 调试失败的动作
                                  - 验证结果
                                  - 状态图构建
                                  默认: "./screenshots"

        配置存储:
            这些值被存储并在 start() 调用时使用。
            浏览器启动后无法更改。

        示例:
            >>> # 开发模式（可见浏览器）
            >>> browser = PlaywrightBrowser(headless=False)

            >>> # 生产模式（不可见）
            >>> browser = PlaywrightBrowser(headless=True, timeout=60000)

            >>> # 自定义截图位置
            >>> browser = PlaywrightBrowser(screenshot_dir="./debug_screenshots")
        """
        # 存储配置
        self.headless = headless
        self.timeout = timeout
        self.screenshot_dir = Path(screenshot_dir)

        # 浏览器组件（在 start() 中初始化）
        # None 表示浏览器尚未启动
        self.playwright = None       # Playwright 库实例
        self.browser: Optional[Browser] = None   # Chromium 浏览器
        self.context: Optional[BrowserContext] = None  # 隔离会话
        self.page: Optional[Page] = None  # 当前网页

        # 状态追踪
        self.current_state = BrowserState()  # 当前浏览器信息
        self.step_count = 0  # 截图命名的计数器

        logger.info(f"PlaywrightBrowser 已初始化 (headless={headless})")

    # ------------------------------------------------------------------------
    # 浏览器生命周期方法
    # ------------------------------------------------------------------------

    async def start(self) -> None:
        """
        启动浏览器并创建必要的组件

        该方法执行完整的浏览器初始化序列:
            1. 如需要则创建截图目录
            2. 启动 Playwright 库
            3. 启动 Chromium 浏览器
            4. 创建浏览器上下文（隔离会话）
            5. 创建新页面
            6. 设置默认超时

        浏览器上下文说明:
            上下文就像独立的浏览器配置/会话:
            - 有自己的 cookies、缓存、localStorage
            - 可以同时存在多个上下文
            - 用于多账号场景

        浏览器启动参数:
            --disable-blink-features=AutomationControlled:
                隐藏自动化签名（反机器人检测）
            --no-sandbox:
                某些 Docker/Linux 环境需要
            --disable-dev-shm-usage:
                防止容器中的内存问题

        User Agent 设置:
            自定义 user agent 使浏览器看起来像真实 Chrome，
            帮助在受保护网站避免机器人检测。

        错误处理:
            如果 start() 失败，后续操作会抛出
            RuntimeError("浏览器未启动")。

        示例:
            >>> browser = PlaywrightBrowser()
            >>> await browser.start()
            >>> # 浏览器现在可以执行操作
            >>> await browser.close()  # 总是清理
        """
        # 防止重复初始化
        if self.browser is not None:
            logger.warning("浏览器已启动")
            return

        # 创建截图目录
        # parents=True: 如需要则创建父目录
        # exist_ok=True: 目录已存在时不报错
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # 启动 Playwright 库
        # async_playwright() 返回上下文管理器，调用 start() 获取 Playwright 实例
        self.playwright = await async_playwright().start()

        # 启动 Chromium 浏览器
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                # 隐藏自动化签名（反机器人）
                '--disable-blink-features=AutomationControlled',
                # 沙箱设置（用于 Docker/Linux）
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )

        # 创建浏览器上下文（隔离会话）
        # viewport: 渲染的窗口大小
        # user_agent: 浏览器识别字符串
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},  # 标准桌面尺寸
            # 真实 Chrome user agent 避免机器人检测
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        # 创建新页面（像打开新标签页）
        self.page = await self.context.new_page()

        # 设置所有操作的默认超时
        self.page.set_default_timeout(self.timeout)

        logger.success("浏览器启动成功")

    async def close(self) -> None:
        """
        关闭浏览器并清理所有资源

        清理序列:
            1. 关闭浏览器（所有页面和上下文）
            2. 停止 Playwright 库
            3. 将所有引用重置为 None

        资源管理:
            正确的清理防止:
            - 僵尸浏览器进程的内存泄漏
            - 未关闭连接的端口冲突
            - 任务管理器中的僵尸 Chromium 实例

        最佳实践:
            即使发生错误也要在操作后调用 close():
            ```python
            try:
                await browser.start()
                await browser.do_something()
            finally:
                await browser.close()  # 总是执行
            ```

        示例:
            >>> await browser.start()
            >>> await browser.open_url("https://example.com")
            >>> await browser.close()  # 清理
        """
        # 如果正在运行则关闭浏览器
        if self.browser:
            await self.browser.close()
            self.browser = None

        # 停止 Playwright 库
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        # 重置所有引用
        self.page = None
        self.context = None

        logger.info("浏览器已关闭")

    # ------------------------------------------------------------------------
    # 页面导航方法
    # ------------------------------------------------------------------------

    async def open_url(self, url: str) -> str:
        """
        在浏览器中打开 URL 并等待页面加载

        该方法:
            1. 导航到 URL
            2. 等待 'networkidle'（页面完全加载）
            3. 更新浏览器状态（URL、标题）
            4. 拍摄初始截图

        Network Idle 说明:
            'networkidle' 意味着:
            - 500ms 内无网络活动
            - 所有关键资源已加载
            - JavaScript 大部分完成
            其他选项: 'load', 'domcontentloaded'

        参数:
            url (str): 要打开的 URL
                       必须是带协议的有效 URL
                       示例: "https://www.baidu.com"

        返回:
            str: 包含 URL 和页面标题的结果消息
                 格式: "已打开页面: {url}\n标题: {title}"

        状态更新:
            - current_state.url: 设置为打开的 URL
            - current_state.title: 设置为页面标题

        示例:
            >>> result = await browser.open_url("https://www.baidu.com")
            >>> print(result)
            "已打开页面: https://www.baidu.com\n标题: 百度一下"
        """
        if not self.page:
            raise RuntimeError("浏览器未启动。请先调用 start()")

        logger.info(f"正在打开 URL: {url}")

        # 导航到 URL
        # wait_until='networkidle': 等待页面完全加载
        await self.page.goto(url, wait_until='networkidle')

        # 更新追踪的状态
        self.current_state.start_url = url
        # 自动识别网站名称（从URL提取）
        if "baidu" in url:
            self.current_state.website = "百度"
        elif "douban" in url:
            self.current_state.website = "豆瓣"
        elif "github" in url:
            self.current_state.website = "GitHub"
        elif "amazon" in url:
            self.current_state.website = "亚马逊"
        else:
            # 从域名提取网站名
            domain = url.split("//")[1].split("/")[0].split(".")[0]
            self.current_state.website = domain

        # 拍摄初始截图用于状态追踪
        await self._save_screenshot("page_opened")

        logger.info(f"页面已打开: {url}")

        return f"已打开页面: {url}\n网站: {self.current_state.website}"

    # ------------------------------------------------------------------------
    # 元素交互方法
    # ------------------------------------------------------------------------

    async def click(self, selector: str) -> str:
        """
        使用 CSS 选择器点击页面上的元素

        CSS 选择器示例:
            - "#submit": id="submit" 的元素
            - ".btn-primary": class="btn-primary" 的元素
            - "button[type='submit']": 有 type 属性的按钮
            - "div > a:first-child": div 中第一个链接

        参数:
            selector (str): 用于查找元素的 CSS 选择器
                            必须精确匹配一个可见元素

        返回:
            str: 结果消息
                 成功: "已点击元素: {selector}"
                 失败: "点击失败: {错误消息}"

        行为:
            - 最多等待 5 秒让元素出现
            - 点击元素
            - 等待 0.5 秒用于可能的导航
            - 拍摄截图用于状态追踪

        常见错误:
            - 元素未找到: 选择器不匹配任何内容
            - 元素隐藏: 元素存在但不可见
            - 多个元素: 选择器匹配太多元素

        示例:
            >>> # 点击百度搜索按钮
            >>> result = await browser.click("#su")
            >>> print(result)
            "已点击元素: #su"
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        logger.info(f"正在点击元素: {selector}")

        try:
            # 等待元素存在于DOM（不要求可见）
            await self.page.wait_for_selector(selector, timeout=5000, state="attached")

            # 点击元素（force=True 强制点击隐藏元素）
            await self.page.click(selector, force=True)

            # 等待可能的页面变化
            # （导航、加载、JavaScript 动作）
            await asyncio.sleep(0.5)

            # 更新步骤计数器并拍摄截图
            self.step_count += 1
            await self._save_screenshot(f"click_{self.step_count}")

            return f"已点击元素: {selector}"

        except Exception as e:
            logger.error(f"点击失败: {e}")
            return f"点击失败: {str(e)}"

    async def type_text(self, selector: str, text: str) -> str:
        """
        在输入框中输入文字

        该方法:
            1. 等待输入元素出现
            2. 清除已有文字（如果有）
            3. 逐字符输入新文字
            4. 拍摄截图用于验证

        参数:
            selector (str): 输入框的 CSS 选择器
                            示例: "#kw", "input[name='search']"

            text (str): 要输入的文字内容
                        会逐字符输入

        返回:
            str: 结果消息
                 成功: "已输入文字 '{text}' 到 {selector}"
                 失败: "输入失败: {错误}"

        fill() vs type() 的区别:
            fill(): 清除字段并立即设置文字
            type(): 逐字符输入（更真实）
            我们使用 fill() 因为简单可靠

        示例:
            >>> # 在百度输入搜索词
            >>> result = await browser.type_text("#kw", "Python教程")
            >>> print(result)
            "已输入文字 'Python教程' 到 #kw"
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        logger.info(f"正在输入 '{text}' 到: {selector}")

        try:
            # 等待输入元素存在于DOM（不要求可见）
            await self.page.wait_for_selector(selector, timeout=5000, state="attached")

            # 清除已有文字并填入新文字（force=True 强制操作隐藏元素）
            await self.page.fill(selector, text, force=True)

            # 更新步骤计数器并截图
            self.step_count += 1
            await self._save_screenshot(f"type_{self.step_count}")

            return f"已输入文字 '{text}' 到 {selector}"

        except Exception as e:
            logger.error(f"输入失败: {e}")
            return f"输入失败: {str(e)}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """
        垂直滚动页面

        滚动用途:
        - 加载懒加载内容（无限滚动）
        - 查找视口下方的元素
        - 查看完整页面内容

        参数:
            direction (str): 滚动方向
                             选项: "up"（上）或 "down"（下）
                             默认: "down"

            amount (int): 滚动的像素数
                          默认: 500（适度滚动）
                          更大值 = 更大滚动

        返回:
            str: 结果消息
                 "已滚动页面 {direction} {amount}px"

        实现:
            使用 JavaScript window.scrollBy() 实现可靠滚动

        示例:
            >>> # 向下滚动加载更多内容
            >>> result = await browser.scroll("down", 800)
            >>> print(result)
            "已滚动页面 down 800px"
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        logger.info(f"正在滚动 {direction} {amount}px")

        # 计算滚动距离
        delta_y = amount if direction == "down" else -amount

        # 执行 JavaScript 滚动
        await self.page.evaluate(f"window.scrollBy(0, {delta_y})")

        # 等待滚动完成
        await asyncio.sleep(0.3)

        # 追踪步骤并拍摄截图
        self.step_count += 1
        await self._save_screenshot(f"scroll_{self.step_count}")

        return f"已滚动页面 {direction} {amount}px"

    async def wait(self, seconds: float = 1.0) -> str:
        """
        等待指定时间

        等待用途:
        - 让动画完成
        - 等待 AJAX 内容加载
        - 避免速率限制

        参数:
            seconds (float): 等待的秒数
                             默认: 1.0 秒

        返回:
            str: 结果消息 "已等待 {seconds} 秒"

        注意:
            这是简单的 sleep，不是智能等待
            对于等待特定条件，请使用:
            - page.wait_for_selector()
            - page.wait_for_load_state()

        示例:
            >>> await browser.wait(2.0)  # 等待2秒
        """
        logger.info(f"正在等待 {seconds} 秒")
        await asyncio.sleep(seconds)
        return f"已等待 {seconds} 秒"

    # ------------------------------------------------------------------------
    # 内容提取方法
    # ------------------------------------------------------------------------

    async def extract_text(self, selector: str = "body") -> str:
        """
        从元素提取文本内容

        该方法从任何元素获取可见文本，
        用于为 LLM 上下文获取页面内容。

        参数:
            selector (str): 目标元素的 CSS 选择器
                            默认: "body"（整个页面）
                            示例: "#content", ".article", "h1"

        返回:
            str: 提取的文本内容
                 如果太长则截断到 1000 字符
                 格式: "{text}...[截断]" 如果截断
                 错误: "未找到元素: {selector}" 如果未找到

        截断逻辑:
            LLM 上下文窗口有限
            截断防止过多的 token 使用
            1000 字符 ≈ 500 tokens（粗略估计）

        示例:
            >>> # 获取整个页面的文本
            >>> text = await browser.extract_text()
            >>> print(text[:100])  # 前100字符

            >>> # 获取特定元素的文本
            >>> title = await browser.extract_text("h1")
            >>> print(title)
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        logger.info(f"正在从 {selector} 提取文本")

        try:
            # 查询元素
            element = await self.page.query_selector(selector)

            if element:
                # 获取内部文本（可见文本内容）
                text = await element.inner_text()

                # 如果太长则截断（节省内存）
                if len(text) > 1000:
                    text = text[:1000] + "...[截断]"

                # 存储到完成界面描述
                self.current_state.completion_interface = text[:200]

                return text
            else:
                return f"未找到元素: {selector}"

        except Exception as e:
            logger.error(f"提取文本失败: {e}")
            return f"提取失败: {str(e)}"

    async def find_elements(self, selector: str) -> List[Dict[str, str]]:
        """
        查找所有匹配 CSS 选择器的元素

        该方法返回元素信息列表，
        用于查找多个相似元素（链接、按钮等）。

        参数:
            selector (str): 要匹配的 CSS 选择器
                            示例: "a"（所有链接）, ".item"（所有项目）

        返回:
            List[Dict[str, str]]: 元素信息列表
                每个字典包含:
                - index: 列表中的位置（0, 1, 2, ...）
                - text: 元素的文本内容（截断）
                - href: 链接 URL（如果元素是 <a>）

        限制:
            最多返回 10 个元素以防止内存问题

        示例:
            >>> # 查找页面上的所有链接
            >>> links = await browser.find_elements("a")
            >>> for link in links:
            ...     print(f"{link['index']}: {link['text']}")
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        # 查询所有匹配的元素
        elements = await self.page.query_selector_all(selector)
        results = []

        # 从每个元素提取信息（限制为10个）
        for i, element in enumerate(elements[:10]):
            # 获取文本内容
            text = await element.inner_text()

            # 获取 href 属性（用于链接）
            href = await element.get_attribute('href') or ''

            results.append({
                "index": i,
                "text": text[:100],  # 截断
                "href": href
            })

        return results

    # ------------------------------------------------------------------------
    # 视觉捕获方法
    # ------------------------------------------------------------------------

    async def screenshot(self, name: str = None) -> str:
        """
        拍摄当前页面的截图

        截图用途:
        - 调试: 看到代理看到的
        - 验证: 确认任务完成
        - 状态图: 视觉状态追踪

        参数:
            name (str): 可选的文件名前缀
                        默认: "screenshot_{step_count}"

        返回:
            str: 包含文件路径的结果消息
                 "截图已保存: {path}"

        文件命名:
            格式: "{name}_{timestamp}.png"
            时间戳确保文件名唯一

        示例:
            >>> # 拍摄命名截图
            >>> result = await browser.screenshot("final_result")
            >>> print(result)
            "截图已保存: ./screenshots/final_result_1234567890.png"
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        filename = name or f"screenshot_{self.step_count}"
        path = await self._save_screenshot(filename)

        return f"截图已保存: {path}"

    async def _save_screenshot(self, name: str) -> str:
        """
        保存截图到磁盘（纯 Playwright PNG 格式）

        参数:
            name (str): 文件名前缀

        返回:
            str: 保存截图的完整路径
        """
        if not self.page:
            return ""

        timestamp = asyncio.get_event_loop().time()
        filename = f"{name}_{int(timestamp)}.png"
        path = self.screenshot_dir / filename

        await self.page.screenshot(path=str(path))

        # 更新状态中的截图路径
        self.current_state.screenshot_path = str(path)
        # 更新当前 URL 和标题
        self.current_state.url = self.page.url
        try:
            self.current_state.title = await self.page.title()
        except Exception:
            self.current_state.title = ""

        logger.debug(f"截图已保存: {path}")

        return str(path)

    # ------------------------------------------------------------------------
    # 键盘方法
    # ------------------------------------------------------------------------

    async def press_key(self, key: str) -> str:
        """
        按下键盘按键

        用途:
        - 提交表单（Enter）
        - 取消操作（Escape）
        - 导航（Tab、方向键）

        参数:
            key (str): 要按下的按键
                       示例: "Enter", "Escape", "Tab"
                       特殊按键有名称，普通按键就是字符

        返回:
            str: 结果消息 "已按下按键: {key}"

        常用按键:
            - Enter: 提交表单、确认动作
            - Escape: 取消、关闭对话框
            - Tab: 移动焦点到下一个元素
            - ArrowUp/Down: 导航

        示例:
            >>> # 提交搜索表单
            >>> await browser.type_text("#search", "query")
            >>> await browser.press_key("Enter")
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        await self.page.keyboard.press(key)
        logger.info(f"已按下按键: {key}")

        return f"已按下按键: {key}"

    # ------------------------------------------------------------------------
    # 状态方法
    # ------------------------------------------------------------------------

    async def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前浏览器状态的字典

        返回:
            Dict[str, Any]: 状态字典
        """
        if not self.page:
            return {"error": "浏览器未启动"}

        current_url = self.page.url
        page_title = await self.page.title()

        # 更新状态
        self.current_state.url = current_url
        self.current_state.title = page_title
        self.current_state.completion_interface = f"页面标题: {page_title}"

        return {
            "website": self.current_state.website,
            "start_url": self.current_state.start_url,
            "url": self.current_state.url,
            "title": self.current_state.title,
            "prompt": self.current_state.prompt,
            "completion_interface": self.current_state.completion_interface,
            "screenshot_path": self.current_state.screenshot_path,
        }

    # ------------------------------------------------------------------------
    # 页面分析方法（让 Agent 能看见页面）
    # ------------------------------------------------------------------------

    async def get_interactive_elements(self) -> List[Dict[str, str]]:
        """
        获取页面上所有可交互元素（输入框、按钮、链接等）

        返回元素的 selector、类型、文本等信息，供 LLM 决策使用
        """
        if not self.page:
            return []

        elements = []
        selectors = [
            ("input[type='text']", "text_input"),
            ("input[type='search']", "search_input"),
            ("input:not([type])", "text_input"),
            ("button", "button"),
            ("[role='button']", "button"),
            ("a[href]", "link"),
            ("[onclick]", "clickable"),
            ("select", "dropdown"),
        ]

        for selector, elem_type in selectors:
            items = await self.page.query_selector_all(selector)
            for i, elem in enumerate(items[:5]):  # 每类最多5个
                try:
                    # 获取元素属性
                    id_attr = await elem.get_attribute("id") or ""
                    class_attr = await elem.get_attribute("class") or ""
                    text = await elem.inner_text() if elem_type in ["button", "link", "clickable"] else ""
                    placeholder = await elem.get_attribute("placeholder") or ""
                    name_attr = await elem.get_attribute("name") or ""

                    # 构建 selector
                    best_selector = ""
                    if id_attr:
                        best_selector = f"#{id_attr}"
                    elif name_attr:
                        best_selector = f"[name='{name_attr}']"
                    elif class_attr:
                        first_class = class_attr.split()[0]
                        best_selector = f".{first_class}"
                    else:
                        best_selector = selector

                    elements.append({
                        "selector": best_selector,
                        "type": elem_type,
                        "id": id_attr,
                        "class": class_attr[:50],
                        "text": text[:50],
                        "placeholder": placeholder[:50],
                        "name": name_attr,
                    })
                except:
                    continue

        return elements

    # ------------------------------------------------------------------------
    # 高级方法
    # ------------------------------------------------------------------------

    async def execute_script(self, script: str) -> Any:
        """
        在浏览器中执行 JavaScript

        用于:
        - 高级 DOM 操作
        - 自定义数据提取
        - 标准方法无法覆盖的复杂交互

        参数:
            script (str): 要执行的 JavaScript 代码
                          可以返回值

        返回:
            Any: JavaScript 的返回值（如果有）

        安全提示:
            谨慎使用 - 可以任意修改页面

        示例:
            >>> # 获取所有链接 URL
            >>> links = await browser.execute_script(
            ...     "Array.from(document.querySelectorAll('a')).map(a => a.href)"
            ... )
        """
        if not self.page:
            raise RuntimeError("浏览器未启动")

        return await self.page.evaluate(script)