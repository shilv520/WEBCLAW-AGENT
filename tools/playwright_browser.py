"""
Playwright Browser Tool - Core Browser Automation Engine

This module provides a comprehensive Playwright-based browser automation
tool that enables the agent to interact with web pages like a human user.

Key Features:
- Page Navigation: Open URLs, navigate pages
- Element Interaction: Click, type, scroll, press keys
- Content Extraction: Get text, HTML, element info
- Visual Capture: Screenshots for debugging and verification
- State Tracking: Maintain current browser state

Architecture:
    ┌─────────────────┐
    │  ExecutorAgent  │
    └─────────────────┘
           │ calls
           ↓
    ┌─────────────────┐
    │ PlaywrightBrowser│
    │  ┌───────────┐  │
    │  │ Playwright│  │ ← Browser automation library
    │  │  Chromium │  │ ← Browser instance
    │  │   Page    │  │ ← Web page context
    │  └───────────┘  │
    └─────────────────┘
           │
           ↓
    ┌─────────────────┐
    │   Web Page      │
    │ (target site)   │
    └─────────────────┘

Async Programming Note:
    All methods are async (using async/await) because:
    - Browser operations are I/O bound (network, rendering)
    - Async allows non-blocking execution
    - Multiple operations can run concurrently

Author: WebClaw Team
Version: 0.1.0
"""

import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger
from pydantic import BaseModel

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# ============================================================================
# BROWSER STATE MODEL
# ============================================================================

class BrowserState(BaseModel):
    """
    Pydantic model representing the current browser state.

    This model is used to track and validate browser state changes.
    Pydantic provides automatic validation and serialization.

    Attributes:
        url (str): Current page URL.
                   Empty if no page loaded.
                   Example: "https://www.baidu.com"

        title (str): Page title from <title> tag.
                     Example: "百度一下，你就知道"

        content (str): Text content preview from page body.
                       Truncated for memory efficiency.
                       Used for LLM context.

        screenshot_path (str): Path to most recent screenshot.
                               Empty if no screenshot taken.
                               Used for verification and debugging.

    Example:
        >>> state = BrowserState(
        ...     url="https://www.baidu.com",
        ...     title="百度一下"
        ... )
        >>> print(state.url)
        "https://www.baidu.com"
    """
    url: str = ""
    title: str = ""
    content: str = ""
    screenshot_path: str = ""


# ============================================================================
# PLAYWRIGHT BROWSER CLASS
# ============================================================================

class PlaywrightBrowser:
    """
    Playwright-based browser automation tool for web agent execution.

    This class wraps Playwright's async API to provide a clean, agent-friendly
    interface for browser operations. It handles:
    - Browser lifecycle (start/close)
    - Page navigation and interaction
    - Content extraction
    - Visual capture (screenshots)
    - State tracking

    Usage Pattern:
        1. Initialize with config (headless, timeout, etc.)
        2. Call start() to launch browser
        3. Perform operations (open_url, click, type, etc.)
        4. Call close() to cleanup resources

    Key Design Decisions:
        - Async methods: Non-blocking I/O for better performance
        - Automatic screenshots: Capture state for debugging
        - Error handling: Return error messages instead of raising
        - State tracking: Maintain BrowserState for context

    Attributes:
        headless (bool): Run without visible browser window
        timeout (int): Default operation timeout in milliseconds
        screenshot_dir (Path): Directory for saving screenshots

        playwright: Playwright instance (from async_playwright)
        browser: Chromium browser instance
        context: Browser context (isolated session)
        page: Current web page

        current_state: Tracked browser state
        step_count: Counter for naming screenshots

    Example:
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
        Initialize the browser tool with configuration.

        Args:
            headless (bool): Browser visibility mode.
                             False: Show browser window (for debugging)
                             True: Run invisibly (for production)
                             Default: False (recommended for development)

            timeout (int): Default timeout for operations in milliseconds.
                           Network operations will fail after this time.
                           Default: 30000 (30 seconds)
                           Increase for slow networks/sites

            screenshot_dir (str): Directory path for saving screenshots.
                                  Screenshots are used for:
                                  - Debugging failed actions
                                  - Verification of results
                                  - State graph construction
                                  Default: "./screenshots"

        Configuration Storage:
            These values are stored and used when start() is called.
            They cannot be changed after browser starts.

        Example:
            >>> # Development mode (visible browser)
            >>> browser = PlaywrightBrowser(headless=False)

            >>> # Production mode (invisible)
            >>> browser = PlaywrightBrowser(headless=True, timeout=60000)

            >>> # Custom screenshot location
            >>> browser = PlaywrightBrowser(screenshot_dir="./debug_screenshots")
        """
        # Store configuration
        self.headless = headless
        self.timeout = timeout
        self.screenshot_dir = Path(screenshot_dir)

        # Browser components (initialized in start())
        # None indicates browser not started yet
        self.playwright = None       # Playwright library instance
        self.browser: Optional[Browser] = None   # Chromium browser
        self.context: Optional[BrowserContext] = None  # Isolated session
        self.page: Optional[Page] = None  # Current web page

        # State tracking
        self.current_state = BrowserState()  # Current browser info
        self.step_count = 0  # Counter for screenshot naming

        logger.info(f"PlaywrightBrowser initialized (headless={headless})")

    # ------------------------------------------------------------------------
    # Browser Lifecycle Methods
    # ------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the browser and create necessary components.

        This method performs the full browser initialization sequence:
            1. Create screenshot directory if needed
            2. Launch Playwright library
            3. Launch Chromium browser
            4. Create browser context (isolated session)
            5. Create new page
            6. Set default timeout

        Browser Context Explanation:
            A context is like a separate browser profile/session:
            - Has its own cookies, cache, localStorage
            - Multiple contexts can exist simultaneously
            - Useful for multi-account scenarios

        Browser Launch Args:
            --disable-blink-features=AutomationControlled:
                Hides automation signature (anti-bot detection)
            --no-sandbox:
                Needed for some Docker/Linux environments
            --disable-dev-shm-usage:
                Prevents memory issues in containers

        User Agent Setting:
            Custom user agent makes browser look like real Chrome,
            helping avoid bot detection on protected sites.

        Error Handling:
            If start() fails, subsequent operations will raise
            RuntimeError("Browser not started").

        Example:
            >>> browser = PlaywrightBrowser()
            >>> await browser.start()
            >>> # Browser is now ready for operations
            >>> await browser.close()  # Always cleanup
        """
        # Prevent duplicate initialization
        if self.browser is not None:
            logger.warning("Browser already started")
            return

        # Create screenshot directory
        # parents=True: Create parent directories if needed
        # exist_ok=True: Don't error if directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Launch Playwright library
        # async_playwright() returns a context manager
        self.playwright = await async_playwright.start()

        # Launch Chromium browser
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                # Hide automation signature (anti-bot)
                '--disable-blink-features=AutomationControlled',
                # Sandbox settings (for Docker/Linux)
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )

        # Create browser context (isolated session)
        # viewport: Window size for rendering
        # user_agent: Browser identification string
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},  # Standard desktop size
            # Real Chrome user agent to avoid bot detection
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        # Create new page (like opening a new tab)
        self.page = await self.context.new_page()

        # Set default timeout for all operations
        self.page.set_default_timeout(self.timeout)

        logger.success("Browser started successfully")

    async def close(self) -> None:
        """
        Close the browser and cleanup all resources.

        Cleanup Sequence:
            1. Close browser (all pages and contexts)
            2. Stop Playwright library
            3. Reset all references to None

        Resource Management:
            Proper cleanup prevents:
            - Memory leaks from zombie browser processes
            - Port conflicts from unclosed connections
            - Zombie Chromium instances in task manager

        Best Practice:
            Always call close() after operations, even if errors occur:
            ```python
            try:
                await browser.start()
                await browser.do_something()
            finally:
                await browser.close()  # Always executed
            ```

        Example:
            >>> await browser.start()
            >>> await browser.open_url("https://example.com")
            >>> await browser.close()  # Cleanup
        """
        # Close browser if running
        if self.browser:
            await self.browser.close()
            self.browser = None

        # Stop Playwright library
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        # Reset all references
        self.page = None
        self.context = None

        logger.info("Browser closed")

    # ------------------------------------------------------------------------
    # Page Navigation Methods
    # ------------------------------------------------------------------------

    async def open_url(self, url: str) -> str:
        """
        Open a URL in the browser and wait for page load.

        This method:
            1. Navigates to the URL
            2. Waits for 'networkidle' (page fully loaded)
            3. Updates browser state (URL, title)
            4. Takes initial screenshot

        Network Idle Explanation:
            'networkidle' means:
            - No network activity for 500ms
            - All critical resources loaded
            - JavaScript mostly complete
            Alternative options: 'load', 'domcontentloaded'

        Args:
            url (str): The URL to open.
                       Must be a valid URL with protocol.
                       Example: "https://www.baidu.com"

        Returns:
            str: Result message with URL and page title.
                 Format: "已打开页面: {url}\n标题: {title}"

        State Updates:
            - current_state.url: Set to opened URL
            - current_state.title: Set to page title

        Example:
            >>> result = await browser.open_url("https://www.baidu.com")
            >>> print(result)
            "已打开页面: https://www.baidu.com\n标题: 百度一下"
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.info(f"Opening URL: {url}")

        # Navigate to URL
        # wait_until='networkidle': Wait for page to fully load
        await self.page.goto(url, wait_until='networkidle')

        # Update tracked state
        self.current_state.url = url
        self.current_state.title = await self.page.title()

        # Take initial screenshot for state tracking
        await self._save_screenshot("page_opened")

        logger.info(f"Page opened: {self.current_state.title}")

        return f"已打开页面: {url}\n标题: {self.current_state.title}"

    # ------------------------------------------------------------------------
    # Element Interaction Methods
    # ------------------------------------------------------------------------

    async def click(self, selector: str) -> str:
        """
        Click an element on the page using CSS selector.

        CSS Selector Examples:
            - "#submit": Element with id="submit"
            - ".btn-primary": Element with class="btn-primary"
            - "button[type='submit']": Button with type attribute
            - "div > a:first-child": First anchor in div

        Args:
            selector (str): CSS selector to find the element.
                            Must match exactly one visible element.

        Returns:
            str: Result message.
                 Success: "已点击元素: {selector}"
                 Failure: "点击失败: {error message}"

        Behavior:
            - Waits up to 5 seconds for element to appear
            - Clicks the element
            - Waits 0.5s for potential navigation
            - Takes screenshot for state tracking

        Common Errors:
            - Element not found: Selector doesn't match anything
            - Element hidden: Element exists but not visible
            - Multiple elements: Selector matches too many

        Example:
            >>> # Click search button on Baidu
            >>> result = await browser.click("#su")
            >>> print(result)
            "已点击元素: #su"
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        logger.info(f"Clicking element: {selector}")

        try:
            # Wait for element to appear (5 second timeout)
            await self.page.wait_for_selector(selector, timeout=5000)

            # Click the element
            await self.page.click(selector)

            # Wait for potential page changes
            # (navigation, loading, JavaScript actions)
            await asyncio.sleep(0.5)

            # Update step counter and take screenshot
            self.step_count += 1
            await self._save_screenshot(f"click_{self.step_count}")

            return f"已点击元素: {selector}"

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return f"点击失败: {str(e)}"

    async def type_text(self, selector: str, text: str) -> str:
        """
        Type text into an input field.

        This method:
            1. Waits for input element to appear
            2. Clears existing text (if any)
            3. Types new text character by character
            4. Takes screenshot for verification

        Args:
            selector (str): CSS selector for the input field.
                            Examples: "#kw", "input[name='search']"

            text (str): Text content to type.
                        Will be typed character by character.

        Returns:
            str: Result message.
                 Success: "已输入文字 '{text}' 到 {selector}"
                 Failure: "输入失败: {error}"

        fill() vs type():
            fill(): Clears field and sets text instantly
            type(): Types character by character (more realistic)
            We use fill() for simplicity and reliability.

        Example:
            >>> # Type search query in Baidu
            >>> result = await browser.type_text("#kw", "Python教程")
            >>> print(result)
            "已输入文字 'Python教程' 到 #kw"
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        logger.info(f"Typing '{text}' into: {selector}")

        try:
            # Wait for input element
            await self.page.wait_for_selector(selector, timeout=5000)

            # Clear existing text and fill with new text
            # fill() is faster and more reliable than type()
            await self.page.fill(selector, text)

            # Update step counter and screenshot
            self.step_count += 1
            await self._save_screenshot(f"type_{self.step_count}")

            return f"已输入文字 '{text}' 到 {selector}"

        except Exception as e:
            logger.error(f"Type failed: {e}")
            return f"输入失败: {str(e)}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """
        Scroll the page vertically.

        Scrolling is needed for:
        - Loading lazy-loaded content (infinite scroll)
        - Finding elements below viewport
        - Viewing full page content

        Args:
            direction (str): Scroll direction.
                             Options: "up" or "down"
                             Default: "down"

            amount (int): Pixels to scroll.
                          Default: 500 (moderate scroll)
                          Larger values = bigger scroll

        Returns:
            str: Result message.
                 "已滚动页面 {direction} {amount}px"

        Implementation:
            Uses JavaScript window.scrollBy() for reliable scrolling.

        Example:
            >>> # Scroll down to load more content
            >>> result = await browser.scroll("down", 800)
            >>> print(result)
            "已滚动页面 down 800px"
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        logger.info(f"Scrolling {direction} by {amount}px")

        # Calculate scroll delta
        delta_y = amount if direction == "down" else -amount

        # Execute JavaScript to scroll
        await self.page.evaluate(f"window.scrollBy(0, {delta_y})")

        # Wait for scroll to complete
        await asyncio.sleep(0.3)

        # Track step and take screenshot
        self.step_count += 1
        await self._save_screenshot(f"scroll_{self.step_count}")

        return f"已滚动页面 {direction} {amount}px"

    async def wait(self, seconds: float = 1.0) -> str:
        """
        Wait for a specified duration.

        Waiting is useful for:
        - Letting animations complete
        - Waiting for AJAX content to load
        - Avoiding rate limiting

        Args:
            seconds (float): Duration to wait in seconds.
                             Default: 1.0 second

        Returns:
            str: Result message "已等待 {seconds} 秒"

        Note:
            This is a simple sleep, not a smart wait.
            For waiting for specific conditions, use:
            - page.wait_for_selector()
            - page.wait_for_load_state()

        Example:
            >>> await browser.wait(2.0)  # Wait 2 seconds
        """
        logger.info(f"Waiting {seconds} seconds")
        await asyncio.sleep(seconds)
        return f"已等待 {seconds} 秒"

    # ------------------------------------------------------------------------
    # Content Extraction Methods
    # ------------------------------------------------------------------------

    async def extract_text(self, selector: str = "body") -> str:
        """
        Extract text content from an element.

        This method retrieves visible text from any element,
        useful for getting page content for LLM context.

        Args:
            selector (str): CSS selector for target element.
                            Default: "body" (entire page)
                            Examples: "#content", ".article", "h1"

        Returns:
            str: Extracted text content.
                 Truncated to 1000 chars if too long.
                 Format: "{text}...[截断]" if truncated
                 Error: "未找到元素: {selector}" if not found

        Truncation Logic:
            LLM context windows are limited.
            Truncating prevents excessive token usage.
            1000 chars ≈ 500 tokens (rough estimate).

        Example:
            >>> # Get entire page text
            >>> text = await browser.extract_text()
            >>> print(text[:100])  # First 100 chars

            >>> # Get specific element text
            >>> title = await browser.extract_text("h1")
            >>> print(title)
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        logger.info(f"Extracting text from: {selector}")

        try:
            # Query for element
            element = await self.page.query_selector(selector)

            if element:
                # Get inner text (visible text content)
                text = await element.inner_text()

                # Truncate if too long (save memory)
                if len(text) > 1000:
                    text = text[:1000] + "...[截断]"

                # Store in state
                self.current_state.content = text

                return text
            else:
                return f"未找到元素: {selector}"

        except Exception as e:
            logger.error(f"Extract text failed: {e}")
            return f"提取失败: {str(e)}"

    async def find_elements(self, selector: str) -> List[Dict[str, str]]:
        """
        Find all elements matching a CSS selector.

        This method returns a list of element information,
        useful for finding multiple similar elements (links, buttons, etc).

        Args:
            selector (str): CSS selector to match.
                            Examples: "a" (all links), ".item" (all items)

        Returns:
            List[Dict[str, str]]: List of element info.
                Each dict contains:
                - index: Position in list (0, 1, 2, ...)
                - text: Element's text content (truncated)
                - href: Link URL (if element is <a>)

        Limit:
            Returns max 10 elements to prevent memory issues.

        Example:
            >>> # Find all links on page
            >>> links = await browser.find_elements("a")
            >>> for link in links:
            ...     print(f"{link['index']}: {link['text']}")
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        # Query all matching elements
        elements = await self.page.query_selector_all(selector)
        results = []

        # Extract info from each element (limit to 10)
        for i, element in enumerate(elements[:10]):
            # Get text content
            text = await element.inner_text()

            # Get href attribute (for links)
            href = await element.get_attribute('href') or ''

            results.append({
                "index": i,
                "text": text[:100],  # Truncate
                "href": href
            })

        return results

    # ------------------------------------------------------------------------
    # Visual Capture Methods
    # ------------------------------------------------------------------------

    async def screenshot(self, name: str = None) -> str:
        """
        Take a screenshot of the current page.

        Screenshots are used for:
        - Debugging: See what agent sees
        - Verification: Confirm task completion
        - State Graph: Visual state tracking

        Args:
            name (str): Optional filename prefix.
                        Default: "screenshot_{step_count}"

        Returns:
            str: Result message with file path.
                 "截图已保存: {path}"

        File Naming:
            Format: "{name}_{timestamp}.png"
            Timestamp ensures unique filenames.

        Example:
            >>> # Take named screenshot
            >>> result = await browser.screenshot("final_result")
            >>> print(result)
            "截图已保存: ./screenshots/final_result_1234567890.png"
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        filename = name or f"screenshot_{self.step_count}"
        path = await self._save_screenshot(filename)

        return f"截图已保存: {path}"

    async def _save_screenshot(self, name: str) -> str:
        """
        Internal method to save screenshot to disk.

        This is a private helper method called by other methods.
        It handles file naming, saving, and state updates.

        Args:
            name (str): Filename prefix.

        Returns:
            str: Full path to saved screenshot.

        File Format:
            {name}_{timestamp}.png
            PNG is chosen for quality and compatibility.
        """
        if not self.page:
            return ""

        # Generate unique filename with timestamp
        timestamp = asyncio.get_event_loop().time()
        filename = f"{name}_{int(timestamp)}.png"
        path = self.screenshot_dir / filename

        # Save screenshot
        await self.page.screenshot(path=str(path))

        # Update state for tracking
        self.current_state.screenshot_path = str(path)
        logger.debug(f"Screenshot saved: {path}")

        return str(path)

    # ------------------------------------------------------------------------
    # Keyboard Methods
    # ------------------------------------------------------------------------

    async def press_key(self, key: str) -> str:
        """
        Press a keyboard key.

        Useful for:
        - Submitting forms (Enter)
        - Canceling operations (Escape)
        - Navigation (Tab, Arrow keys)

        Args:
            key (str): Key to press.
                       Examples: "Enter", "Escape", "Tab"
                       Special keys are named, regular keys are just the character.

        Returns:
            str: Result message "已按下按键: {key}"

        Common Keys:
            - Enter: Submit form, confirm action
            - Escape: Cancel, close dialog
            - Tab: Move focus to next element
            - ArrowUp/Down: Navigation

        Example:
            >>> # Submit search form
            >>> await browser.type_text("#search", "query")
            >>> await browser.press_key("Enter")
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.keyboard.press(key)
        logger.info(f"Pressed key: {key}")

        return f"已按下按键: {key}"

    # ------------------------------------------------------------------------
    # State Methods
    # ------------------------------------------------------------------------

    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current browser state as a dictionary.

        This method provides a complete snapshot of browser state,
        useful for LLM context and state tracking.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - url: Current page URL
                - title: Page title
                - content_preview: Page text (truncated)
                - screenshot_path: Path to latest screenshot

        Use Case:
            LLM needs this info to decide next action.

        Example:
            >>> state = await browser.get_current_state()
            >>> print(f"At: {state['url']}")
            >>> print(f"Title: {state['title']}")
        """
        if not self.page:
            return {"error": "Browser not started"}

        # Get current URL
        url = self.page.url

        # Get page title
        title = await self.page.title()

        # Get body text preview
        body_text = await self.extract_text("body")

        # Update tracked state
        self.current_state.url = url
        self.current_state.title = title
        self.current_state.content = body_text[:500]

        return {
            "url": url,
            "title": title,
            "content_preview": body_text[:500],
            "screenshot_path": self.current_state.screenshot_path
        }

    # ------------------------------------------------------------------------
    # Advanced Methods
    # ------------------------------------------------------------------------

    async def execute_script(self, script: str) -> Any:
        """
        Execute JavaScript in the browser.

        This allows custom JavaScript execution for:
        - Advanced DOM manipulation
        - Custom data extraction
        - Complex interactions not covered by standard methods

        Args:
            script (str): JavaScript code to execute.
                          Can return values.

        Returns:
            Any: Return value from JavaScript (if any).

        Security Note:
            Use with caution - can modify page arbitrarily.

        Example:
            >>> # Get all link URLs
            >>> links = await browser.execute_script(
            ...     "Array.from(document.querySelectorAll('a')).map(a => a.href)"
            ... )
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.evaluate(script)