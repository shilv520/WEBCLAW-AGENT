"""
Week 1 Test Script - Basic functionality tests

Run: python tests/test_week1.py
"""

import asyncio
import pytest
from tools.playwright_browser import PlaywrightBrowser


class TestPlaywrightBrowser:
    """Test basic Playwright browser functionality"""

    @pytest.fixture
    async def browser(self):
        """Create and start browser for tests"""
        b = PlaywrightBrowser(headless=True)
        await b.start()
        yield b
        await b.close()

    @pytest.mark.asyncio
    async def test_start_browser(self):
        """Test browser can start"""
        browser = PlaywrightBrowser(headless=True)
        await browser.start()
        assert browser.browser is not None
        assert browser.page is not None
        await browser.close()

    @pytest.mark.asyncio
    async def test_open_url(self, browser):
        """Test opening a URL"""
        result = await browser.open_url("https://www.baidu.com")
        assert "已打开页面" in result
        assert browser.current_state.url == "https://www.baidu.com"

    @pytest.mark.asyncio
    async def test_screenshot(self, browser):
        """Test taking screenshot"""
        await browser.open_url("https://www.baidu.com")
        result = await browser.screenshot("test")
        assert "截图已保存" in result

    @pytest.mark.asyncio
    async def test_extract_text(self, browser):
        """Test extracting text"""
        await browser.open_url("https://www.baidu.com")
        text = await browser.extract_text("body")
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_type_text(self, browser):
        """Test typing text"""
        await browser.open_url("https://www.baidu.com")
        result = await browser.type_text("#kw", "Python")
        assert "已输入文字" in result

    @pytest.mark.asyncio
    async def test_click(self, browser):
        """Test clicking element"""
        await browser.open_url("https://www.baidu.com")
        await browser.type_text("#kw", "Python")
        result = await browser.click("#su")
        assert "已点击元素" in result

    @pytest.mark.asyncio
    async def test_scroll(self, browser):
        """Test scrolling"""
        await browser.open_url("https://www.baidu.com")
        result = await browser.scroll("down", 500)
        assert "已滚动页面" in result

    @pytest.mark.asyncio
    async def test_get_state(self, browser):
        """Test getting current state"""
        await browser.open_url("https://www.baidu.com")
        state = await browser.get_current_state()
        assert state["url"] == "https://www.baidu.com"
        assert "title" in state


async def manual_test():
    """Manual test for quick verification"""
    print("Running manual test...")

    browser = PlaywrightBrowser(headless=False)

    try:
        await browser.start()

        # Test 1: Open Baidu
        print("Test 1: Opening Baidu...")
        result = await browser.open_url("https://www.baidu.com")
        print(f"  Result: {result}")

        # Test 2: Type search query
        print("Test 2: Typing 'Python'...")
        result = await browser.type_text("#kw", "Python")
        print(f"  Result: {result}")

        # Test 3: Click search button
        print("Test 3: Clicking search button...")
        result = await browser.click("#su")
        print(f"  Result: {result}")

        # Test 4: Wait for results
        print("Test 4: Waiting 2 seconds...")
        await browser.wait(2)

        # Test 5: Take screenshot
        print("Test 5: Taking screenshot...")
        result = await browser.screenshot("search_results")
        print(f"  Result: {result}")

        # Test 6: Get current state
        print("Test 6: Getting current state...")
        state = await browser.get_current_state()
        print(f"  URL: {state['url']}")
        print(f"  Title: {state['title']}")

        print("\nAll manual tests passed!")

    finally:
        await browser.close()


if __name__ == "__main__":
    # Run manual test
    asyncio.run(manual_test())