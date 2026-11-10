import sys, asyncio
from playwright.async_api import async_playwright
port, w, h, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
wait = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": w, "height": h}, has_touch=True)
        await pg.goto(f"http://127.0.0.1:{port}/")
        await pg.wait_for_timeout(wait * 1000)
        await pg.screenshot(path=out)
        await b.close()
asyncio.run(main())
