# More finger cases: staff buttons, edge swipes, two fingers on two cards, fast tap runs.
import asyncio, json, sys, urllib.request
from playwright.async_api import async_playwright
PORT = int(sys.argv[1]); BASE = f"http://127.0.0.1:{PORT}"
def state(): return json.load(urllib.request.urlopen(BASE + "/api/state"))
R = {}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox", "--disable-pinch", "--overscroll-history-navigation=0"])
        ctx = await b.new_context(viewport={"width": 1920, "height": 1080}, has_touch=True)
        pg = await ctx.new_page(); await pg.goto(BASE + "/"); await pg.wait_for_timeout(2500)
        cdp = await ctx.new_cdp_session(pg)
        async def touch(kind, pts):
            await cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": [
                {"x": x, "y": y, "radiusX": 12, "radiusY": 12, "force": 1, "id": i} for i, (x, y) in enumerate(pts)]})
        async def tap(x, y, ms=80):
            await touch("touchStart", [(x, y)]); await pg.wait_for_timeout(ms); await touch("touchEnd", [])
        async def center(sel):
            bb = await pg.locator(sel).bounding_box(); return bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        async def swipe(x0, y0, x1, y1):
            await touch("touchStart", [(x0, y0)])
            for k in range(1, 11):
                await pg.wait_for_timeout(15); await touch("touchMove", [(x0 + (x1 - x0) * k / 10, y0 + (y1 - y0) * k / 10)])
            await touch("touchEnd", []); await pg.wait_for_timeout(500)
        ids = await pg.eval_on_selector_all(".look", "els => els.map(e => e.dataset.id)")
        url0 = pg.url
        # pull down from the top, swipe in from each edge, swipe up from the bottom
        for name, s in {"pull down from top": (900, 5, 900, 700), "swipe from left edge": (2, 500, 900, 500),
                        "swipe from right edge": (1917, 500, 900, 500), "swipe up from bottom": (900, 1075, 900, 300)}.items():
            await swipe(*s)
            R[name + " -> nothing moves"] = (pg.url == url0 and await pg.evaluate("() => scrollX + scrollY + document.scrollingElement.scrollTop") == 0
                                            and await pg.evaluate("() => visualViewport.scale") == 1)
        # double-tap on the video: no zoom
        await tap(500, 500); await pg.wait_for_timeout(60); await tap(500, 500); await pg.wait_for_timeout(500)
        R["double-tap on the video -> no zoom"] = await pg.evaluate("() => visualViewport.scale") == 1
        # double-tap a card: picks it once, no zoom
        before = state()["last_switch_ms"]; t = [i for i in ids if i != state()["active"]][0]
        x, y = await center(f'.look[data-id="{t}"]'); await tap(x, y); await pg.wait_for_timeout(60); await tap(x, y); await pg.wait_for_timeout(900)
        R["double-tap a card -> picks it, no zoom"] = state()["active"] == t and await pg.evaluate("() => visualViewport.scale") == 1
        # two fingers land on two different cards at once: one of them wins, exactly one card lit
        a, c = [i for i in ids if i != state()["active"]][:2]
        (ax, ay), (cx, cy) = await center(f'.look[data-id="{a}"]'), await center(f'.look[data-id="{c}"]')
        await touch("touchStart", [(ax, ay), (cx, cy)]); await pg.wait_for_timeout(80); await touch("touchEnd", []); await pg.wait_for_timeout(1200)
        lit = await pg.eval_on_selector_all('.look[aria-pressed="true"]', "els => els.map(e => e.dataset.id)")
        R["two fingers on two cards -> one filter, one lit card, they agree"] = (state()["active"] in (a, c), lit == [state()["shown"]])
        # a fast run of taps across all four cards ends on the last one
        order = [ids[1], ids[3], ids[0], ids[2], ids[1], ids[3]]
        for i in order:
            x, y = await center(f'.look[data-id="{i}"]'); await tap(x, y, 40); await pg.wait_for_timeout(70)
        await pg.wait_for_timeout(1500)
        lit = await pg.eval_on_selector_all('.look[aria-pressed="true"]', "els => els.map(e => e.dataset.id)")
        R["6 fast taps -> ends on the last tapped"] = state()["active"] == order[-1] and lit == [order[-1]]
        # staff: open by holding, Start calibration by finger, tapping the dim area outside does nothing bad
        tx, ty = await center("#title")
        await touch("touchStart", [(tx, ty)]); await pg.wait_for_timeout(3300); await touch("touchEnd", []); await pg.wait_for_timeout(300)
        R["staff opens by finger"] = await pg.eval_on_selector("#staff", "e => e.open")
        x, y = await center("#cal-start"); await tap(x, y); await pg.wait_for_timeout(600)
        R["Start calibration by finger -> calibrating"] = state()["calibrating"]
        await tap(30, 30); await pg.wait_for_timeout(300)
        R["tap outside the staff box -> box stays, page fine"] = await pg.eval_on_selector("#staff", "e => e.open")
        x, y = await center("#staff-close"); await tap(x, y); await pg.wait_for_timeout(300)
        R["Close by finger"] = not await pg.eval_on_selector("#staff", "e => e.open")
        # cards still work after closing staff
        t = [i for i in ids if i != state()["active"]][0]; x, y = await center(f'.look[data-id="{t}"]'); await tap(x, y); await pg.wait_for_timeout(900)
        R["cards work after staff closes"] = state()["active"] == t
        await b.close()
asyncio.run(main())
for k, v in R.items(): print(f"{k}: {v}")
