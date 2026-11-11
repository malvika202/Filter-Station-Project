# Real-finger touch test: raw touch events (press, wobble, release) through Chrome's input pipeline,
# with the same Chromium flags the Pi uses. Usage: python test_touch.py PORT
import asyncio, json, sys, urllib.request
from playwright.async_api import async_playwright

PORT = int(sys.argv[1]); BASE = f"http://127.0.0.1:{PORT}"
def state(): return json.load(urllib.request.urlopen(BASE + "/api/state"))
R = {}

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=[
            "--no-sandbox", "--disable-pinch", "--overscroll-history-navigation=0", "--touch-events=enabled"])
        ctx = await b.new_context(viewport={"width": 1920, "height": 1080}, has_touch=True, is_mobile=False)
        pg = await ctx.new_page()
        await pg.goto(BASE + "/"); await pg.wait_for_timeout(2500)
        cdp = await ctx.new_cdp_session(pg)

        async def touch(kind, pts):
            await cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": [
                {"x": x, "y": y, "radiusX": 12, "radiusY": 12, "force": 1, "id": i} for i, (x, y) in enumerate(pts)]})

        async def press(x, y, ms, wobble=0, steps=12):
            await touch("touchStart", [(x, y)])
            for k in range(steps):
                await pg.wait_for_timeout(ms // steps)
                dx = wobble * (1 if k % 2 else -1)
                await touch("touchMove", [(x + dx, y + dx / 2)])
            await touch("touchEnd", [])

        async def staff_open(): return await pg.eval_on_selector("#staff", "e => e.open")
        async def close_staff():
            if await staff_open(): await pg.evaluate("() => document.querySelector('#staff').close()")
            await pg.wait_for_timeout(300)

        tb = await pg.locator("#title").bounding_box(); tx, ty = tb["x"] + tb["width"] / 3, tb["y"] + tb["height"] / 2
        for wob in [0, 3, 8, 15, 25]:
            await press(tx, ty, 3300, wobble=wob)
            R[f"title hold 3.3 s, finger wobble {wob}px -> staff opens"] = await staff_open()
            await close_staff()
        await press(tx, ty, 1500); R["title hold 1.5 s -> stays closed"] = not await staff_open(); await close_staff()

        # taps on cards: still, small drag, bigger drag, long press
        ids = await pg.eval_on_selector_all(".look", "els => els.map(e => e.dataset.id)")
        async def card_xy(i):
            bb = await pg.locator(f'.look[data-id="{i}"]').bounding_box(); return bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        async def try_tap(target, ms, wobble):
            x, y = await card_xy(target); await press(x, y, ms, wobble=wobble, steps=3)
            await pg.wait_for_timeout(900); return state()["active"] == target
        cases = [(80, 0), (80, 6), (80, 12), (150, 20), (700, 0), (1200, 4)]
        for n, (ms, wob) in enumerate(cases):
            target = ids[(n + 1) % 4]
            if state()["active"] == target: target = ids[(n + 2) % 4]
            R[f"card tap {ms} ms, drag {wob}px -> switches"] = await try_tap(target, ms, wob)

        # a drag that starts on a card and ends far off it must not pick it
        target = [i for i in ids if i != state()["active"]][0]
        x, y = await card_xy(target)
        await touch("touchStart", [(x, y)])
        for k in range(8):
            await pg.wait_for_timeout(30); await touch("touchMove", [(x - 60 * (k + 1), y)])
        await touch("touchEnd", []); await pg.wait_for_timeout(900)
        R["drag from a card to the video -> does NOT switch"] = state()["active"] != target

        # two-finger pinch and a swipe must not zoom, scroll or navigate
        url0 = pg.url
        await touch("touchStart", [(800, 500), (900, 500)])
        for k in range(10):
            await pg.wait_for_timeout(30); await touch("touchMove", [(800 - 20 * k, 500), (900 + 20 * k, 500)])
        await touch("touchEnd", []); await pg.wait_for_timeout(400)
        R["pinch: page zoom stays 1"] = await pg.evaluate("() => visualViewport.scale")
        await touch("touchStart", [(300, 600)])
        for k in range(10):
            await pg.wait_for_timeout(20); await touch("touchMove", [(300 + 80 * k, 600 - 40 * k)])
        await touch("touchEnd", []); await pg.wait_for_timeout(600)
        R["swipe: no scroll, same page"] = (await pg.evaluate("() => [scrollX, scrollY, document.scrollingElement.scrollTop]"), pg.url == url0)
        # text selection after a long press on text
        await press(tx, ty, 1200)
        R["long press: no text selected"] = await pg.evaluate("() => String(getSelection())") == ""

        # staff buttons by touch
        await press(tx, ty, 3300)
        if await staff_open():
            cb = await pg.locator("#staff-close").bounding_box()
            await press(cb["x"] + cb["width"] / 2, cb["y"] + cb["height"] / 2, 80, steps=2); await pg.wait_for_timeout(300)
            R["staff Close button by touch"] = not await staff_open()
        else:
            R["staff Close button by touch"] = "staff did not open for this test"
        await b.close()

asyncio.run(main())
for k, v in R.items(): print(f"{k}: {v}")
