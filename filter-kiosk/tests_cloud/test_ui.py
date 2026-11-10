# Browser test of the kiosk page in Chrome, with a touchscreen AND a mouse.
# AC1 cards, AC2 taps, AC3 clicks, AC4 the lit card matches the video, AC5 50 switches,
# AC6 card size at several screens, AC8 starts on Goldilocks. Also: staff panel, setup message.
import asyncio, json, random, sys, time, urllib.request
from playwright.async_api import async_playwright

PORT, UNCAL = int(sys.argv[1]), int(sys.argv[2]); REDUCED = sys.argv[3] if len(sys.argv) > 3 else "no-preference"
BASE = f"http://127.0.0.1:{PORT}"
def state(base=BASE): return json.load(urllib.request.urlopen(base + "/api/state"))
results = {}

async def lit(pg):
    return await pg.eval_on_selector_all('.look[aria-pressed="true"]', "els => els.map(e => e.dataset.id)")

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1920, "height": 1080}, has_touch=True, reduced_motion=REDUCED)
        await pg.goto(BASE + "/"); await pg.wait_for_timeout(2500)

        # AC1 + AC8
        cards = await pg.eval_on_selector_all(".look", """els => els.map(e => ({id: e.dataset.id,
            name: e.querySelector('.look-name').textContent, line: e.querySelector('.look-line').textContent,
            art: (e.querySelector('.thumb img') || {}).naturalWidth > 0, disabled: e.disabled}))""")
        results["AC1 cards"] = cards
        results["AC8 lit at start"] = await lit(pg)

        # AC2 to AC5: 50 switches, touch and mouse mixed, with fast double taps and repeats
        rng = random.Random(7); ids = [c["id"] for c in cards]
        times_page, times_server, one_lit_always, ends_right, kinds = [], [], True, 0, {"tap": 0, "click": 0}
        for i in range(50):
            kind = "tap" if i % 2 == 0 else "click"
            target = rng.choice(ids)
            if i % 10 == 3:          # a fast double: tap a different card first, then the target
                other = rng.choice([x for x in ids if x != target])
                await (pg.tap if kind == "tap" else pg.click)(f'.look[data-id="{other}"]')
            if i % 10 == 7:          # tap the card that's already on
                target = (await lit(pg))[0]
            t0 = time.perf_counter()
            await (pg.tap if kind == "tap" else pg.click)(f'.look[data-id="{target}"]')
            kinds[kind] += 1
            # wait until the page AND the kiosk program both show the tapped filter, steady for 200 ms
            settled_at = None
            while time.perf_counter() - t0 < 3:
                now_lit = await lit(pg)
                if len(now_lit) != 1: one_lit_always = False
                agree = now_lit == [target] and state()["shown"] == target
                if agree and settled_at is None: settled_at = time.perf_counter()
                if not agree: settled_at = None
                if settled_at and time.perf_counter() - settled_at > 0.2: break
                await pg.wait_for_timeout(20)
            times_page.append(((settled_at or time.perf_counter()) - t0) * 1000)
            s = state(); good = s["shown"] == target and now_lit == [target]; ends_right += good
            if not good: print(f"  MISS #{i} {kind} target={target} lit={now_lit} shown={s['shown']} waited={(time.perf_counter()-t0)*1000:.0f}ms double={i % 10 == 3} repeat={i % 10 == 7}", flush=True)
            if s["last_switch_ms"] is not None: times_server.append(s["last_switch_ms"])
            await pg.wait_for_timeout(rng.choice([50, 150, 400]))
        results["AC2/3 taps, clicks"] = kinds
        results["AC5 switches ending on the tapped card"] = f"{ends_right}/50"
        results["AC4 exactly one card lit at every check"] = one_lit_always
        results["AC2/3 tap to new picture (server), worst ms"] = max(times_server)
        results["AC2/3 tap to card lit (page), median / worst ms"] = (round(sorted(times_page)[25]), round(max(times_page)))
        # still running? the video keeps moving and is a 640x480 picture
        f1 = state()["frame"]; await pg.wait_for_timeout(1000); f2 = state()["frame"]
        size = await pg.eval_on_selector("#live", "e => [e.naturalWidth, e.naturalHeight]")
        results["video still moving after 50 switches"] = f2 > f1
        results["video size"] = size

        # the sky changes to the new filter's color
        cur = (await lit(pg))[0]; nxt = [x for x in ids if x != cur][0]
        await pg.click(f'.look[data-id="{nxt}"]')
        sky_ok = False
        for _ in range(80):
            if await pg.evaluate("() => document.body.dataset.filter") == nxt: sky_ok = True; break
            await pg.wait_for_timeout(25)
        await pg.wait_for_timeout(2200)
        glow = await pg.evaluate("() => getComputedStyle(document.body).getPropertyValue('--sky-glow').trim()")
        results["sky follows the filter"] = (sky_ok, nxt, glow)

        # staff panel: hold the title for 3 seconds with a finger-sized pointer press
        box = await pg.locator("#title").bounding_box()
        await pg.mouse.move(box["x"] + 40, box["y"] + 20); await pg.mouse.down()
        await pg.wait_for_timeout(1500)
        early = await pg.eval_on_selector("#staff", "e => e.open")
        await pg.wait_for_timeout(1800); await pg.mouse.up()
        opened = await pg.eval_on_selector("#staff", "e => e.open")
        await pg.wait_for_timeout(500)
        facts = await pg.eval_on_selector_all(".staff-facts dd", "els => els.map(e => e.textContent)")
        await pg.screenshot(path="/home/claude/results/ui_staff.png")
        await pg.click("#staff-close")
        results["staff panel: closed at 1.5 s, open at 3.3 s"] = (not early) and opened
        results["staff panel facts"] = facts

        # AC6: smallest card at several screen sizes
        sizes = {}
        for (w, h) in [(1920, 1080), (1280, 720), (1024, 600), (1080, 1920), (720, 1280)]:
            await pg.set_viewport_size({"width": w, "height": h}); await pg.wait_for_timeout(500)
            r = await pg.eval_on_selector_all(".look", "els => els.map(e => { const r = e.getBoundingClientRect(); return [r.width, r.height]; })")
            over = await pg.eval_on_selector_all(".look", "els => els.some(e => e.scrollHeight > e.clientHeight + 1)")
            sizes[f"{w}x{h}"] = (round(min(x[0] for x in r)), round(min(x[1] for x in r)), "text fits" if not over else "TEXT CUT OFF")
        results["AC6 smallest card (w, h) per screen"] = sizes

        # not calibrated yet: the setup message shows
        pg2 = await b.new_page(viewport={"width": 1280, "height": 720}, has_touch=True)
        await pg2.goto(f"http://127.0.0.1:{UNCAL}/"); await pg2.wait_for_timeout(2500)
        results["uncalibrated note"] = await pg2.eval_on_selector("#note", "e => e.hidden ? '(hidden)' : e.textContent")
        await pg2.screenshot(path="/home/claude/results/ui_uncalibrated.png")
        await b.close()

asyncio.run(main())
for k, v in results.items():
    print(f"{k}: {v}")
