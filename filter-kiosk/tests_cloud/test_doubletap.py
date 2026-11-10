# Fast double taps: tap one card, then IMMEDIATELY another. The kiosk must always end on the second.
import asyncio, json, sys, urllib.request
from playwright.async_api import async_playwright
PORT = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1920, "height": 1080}, has_touch=True)
        await pg.goto(f"http://127.0.0.1:{PORT}/"); await pg.wait_for_timeout(2000)
        ids = ["goldilocks", "smile", "clown", "visor"]; wrong = 0
        for i in range(N):
            first, second = ids[i % 4], ids[(i + 1 + i // 4) % 4]
            if first == second: second = ids[(i + 2) % 4]
            # both taps fired without waiting in between, like a quick double tap
            await pg.evaluate(f"""() => {{ document.querySelector('.look[data-id="{first}"]').click();
                                         document.querySelector('.look[data-id="{second}"]').click(); }}""")
            await pg.wait_for_timeout(700)
            s = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state"))
            lit = await pg.eval_on_selector_all('.look[aria-pressed="true"]', "els => els.map(e => e.dataset.id)")
            if s["shown"] != second or lit != [second]: wrong += 1
        print(f"fast double taps that ended on the WRONG card: {wrong}/{N}")
        await b.close()
asyncio.run(main())
