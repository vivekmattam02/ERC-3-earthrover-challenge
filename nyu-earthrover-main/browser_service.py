import os
import time
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")


class BrowserService:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.default_viewport = {"width": 3840, "height": 2160}

    async def initialize_browser(self):
        if not self.browser:
            try:
                self.playwright = await async_playwright().start()
                
                # Use WebKit (Safari's engine) - Chrome/Firefox have WebRTC issues
                # after recent Chromium updates broke WebRTC in headless/automated browsers
                self.browser = await self.playwright.webkit.launch(
                    headless=False,  # WebRTC requires visible browser
                )
                
                self.page = await self.browser.new_page(
                    viewport=self.default_viewport,
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                
                await self.page.goto(
                    "http://127.0.0.1:8000/sdk", 
                    wait_until="networkidle"
                )
                
                await self.page.click("#join")
                
                # Wait for Agora to connect and start streaming
                # This delay is necessary for WebRTC connection establishment
                await self.page.wait_for_timeout(10000)
                
                # Wait for video element
                await self.page.wait_for_selector("video", timeout=60000)
                
                # Wait for video to have data
                await self.page.wait_for_function(
                    """() => {
                        const video = document.querySelector('#player-1000 video');
                        return video && video.readyState >= 2 && video.videoWidth > 0;
                    }""",
                    timeout=30000
                )
                
                await self.page.wait_for_selector("#map", timeout=30000)
                await self.page.set_viewport_size(self.default_viewport)

                await self.page.wait_for_timeout(2000)

                call = f"""() => {{
                    window.initializeImageParams({{
                        imageFormat: "{FORMAT}",
                        imageQuality: {QUALITY}
                    }});
                }}"""
                await self.page.evaluate(call)
                
            except Exception as e:
                print(f"Error initializing browser: {e}")
                await self.close_browser()
                raise

    async def take_screenshot(self, video_output_folder: str, elements: list):
        await self.initialize_browser()

        dimensions = await self.page.evaluate(
            """() => {
            return {
                width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
            }
        }"""
        )

        if (
            dimensions["width"] > self.default_viewport["width"]
            or dimensions["height"] > self.default_viewport["height"]
        ):
            await self.page.set_viewport_size(dimensions)

        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        screenshots = {}
        for name in elements:
            if name in element_map:
                element_id = element_map[name]
                output_path = f"{video_output_folder}/{name}.png"
                element = await self.page.query_selector(element_id)
                if element:
                    start_time = time.time()
                    await element.screenshot(path=output_path)
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000
                    print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                    screenshots[name] = output_path
                else:
                    print(f"Element {element_id} not found")
            else:
                print(f"Invalid element name: {name}")

        return screenshots

    async def data(self) -> dict:
        await self.initialize_browser()

        bot_data = await self.page.evaluate(
            """() => {
        return window.rtm_data;
        }"""
        )

        return bot_data

    async def front(self) -> str:
        await self.initialize_browser()

        try:
            front_frame = await self.page.evaluate(
                """async () => {
                    const frame = await getLastBase64Frame(1000);
                    return frame || null;
                }"""
            )
            return front_frame
        except Exception as e:
            print(f"Error capturing front frame: {e}")
            return None

    async def rear(self) -> str:
        await self.initialize_browser()

        try:
            rear_frame = await self.page.evaluate(
                """async () => {
                    const frame = await getLastBase64Frame(1001);
                    return frame || null;
                }"""
            )
            return rear_frame
        except Exception as e:
            print(f"Error capturing rear frame: {e}")
            return None

    async def send_message(self, message: dict):
        await self.initialize_browser()

        await self.page.evaluate(
            """(message) => {
                window.sendMessage(message);
            }""",
            message,
        )

    async def close_browser(self):
        if self.page:
            await self.page.close()
            self.page = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
