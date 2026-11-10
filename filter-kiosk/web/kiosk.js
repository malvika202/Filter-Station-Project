// Filter Station kiosk page. Talks only to the kiosk program on this Pi (127.0.0.1).
"use strict";

const $ = (s) => document.querySelector(s);
const looks = $("#looks"), screenBox = $("#screen"), live = $("#live"), note = $("#note"), staff = $("#staff");

let filters = [], shown = null, pending = null, lastFrame = -1, lastFrameAt = Date.now();
let engagedSince = null, idleSince = Date.now(), lastState = null;

async function api(path, options) {
  const r = await fetch(path, Object.assign({ cache: "no-store" }, options));
  return r.json();
}

// ---------- the cards ----------
function buildCards(list) {
  filters = list;
  looks.innerHTML = "";
  for (const f of list) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "look";
    b.dataset.id = f.id;
    b.setAttribute("aria-pressed", "false");
    // the picture is a drawn mannequin wearing this filter's real artwork (made by the kiosk)
    b.innerHTML = `<span class="thumb"><img alt="" draggable="false"></span>
      <span class="look-text"><span class="look-name"></span><span class="look-line"></span></span>`;
    const img = b.querySelector(".thumb img");
    img.addEventListener("error", () => { img.hidden = true; });   // missing picture: plain circle
    img.src = `/preview/${f.id}.png`;
    b.querySelector(".look-name").textContent = f.name;
    b.querySelector(".look-line").textContent = f.available ? f.tagline : "Not available right now";
    b.disabled = !f.available;
    // "click" fires for a finger tap on the touchscreen and for a mouse click
    b.addEventListener("click", () => choose(f.id));
    looks.appendChild(b);
  }
}

// Taps are sent one at a time, in the order they happened. (Sent all at once, two fast taps
// could reach the kiosk program in the wrong order and leave the first one on.)
let sending = Promise.resolve();
function choose(id) {
  pending = id;
  paintCards();
  sending = sending
    .then(() => api("/api/filter", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) }))
    .catch(() => { /* the next status check shows what's really on */ })
    .then(() => poll(true));
}

function paintCards() {
  for (const b of looks.children) {
    const id = b.dataset.id;
    // Lit = the filter in the live picture right now (never just the one that was tapped)
    b.setAttribute("aria-pressed", String(id === shown));
    b.classList.toggle("is-pending", id === pending && id !== shown);
  }
}

// ---------- the live view: as big as fits, always 4:3 ----------
function fitScreen() {
  const r = $("#stage").getBoundingClientRect();
  const pad = parseFloat(getComputedStyle($("#stage")).paddingBottom) || 0;
  // a little smaller than the space allows, so the forest shows around it
  const room = innerWidth > innerHeight ? 0.94 : 0.96;
  let h = Math.min((r.height - pad) * room, r.width * 3 / 4 * room);
  h = Math.max(Math.floor(h), 120);
  screenBox.style.height = h + "px";
  screenBox.style.width = Math.floor(h * 4 / 3) + "px";
}

// ---------- status from the kiosk program ----------
function showNote(text) {
  if (text) { note.textContent = text; note.hidden = false; } else { note.hidden = true; }
}

function render(s) {
  lastState = s;
  if (!filters.length || filters.length !== s.filters.length) buildCards(s.filters);
  if (s.shown && s.shown !== shown) {
    shown = s.shown;
    document.body.dataset.filter = shown;      // the sky fades to this filter's color
  }
  if (pending === shown) pending = null;
  paintCards();
  // Right after a tap, check every 0.1 s (not 0.4 s) so the card lights up as soon as it's on
  if (pending !== null) { clearTimeout(render.quick); render.quick = setTimeout(() => poll(true), 100); }

  // The frame around the video takes the sky's color when someone has the filter on
  const now = Date.now();
  if (s.engaged) { engagedSince = engagedSince || now; idleSince = null; }
  else { idleSince = idleSince || now; engagedSince = null; }
  screenBox.classList.toggle("is-lit", engagedSince !== null && now - engagedSince > 500);

  if (s.frame !== lastFrame) { lastFrame = s.frame; lastFrameAt = now; }
  if (!s.camera_ok) showNote("The camera isn't sending video. Check its USB cable.");
  else if (s.calibrating) showNote("Setting up the 5-foot range. Please wait.");
  else if (!s.calibrated) showNote("Setup needed. Staff: hold the Filter Station title for 3 seconds.");
  else if (idleSince !== null && now - idleSince > 1200) showNote("Stand on the spot and face the camera.");
  else if (s.engaged) showNote("");

  if (staff.open) paintStaff(s);
}

async function poll(once) {
  try { render(await api("/api/state")); }
  catch (e) { showNote("Reconnecting to the camera"); }
  if (!once) setTimeout(poll, 400);
}

// The video: one long-lived picture stream. If it ever stops, start it again.
function startStream() { live.src = "/stream.mjpg?t=" + Date.now(); }
live.addEventListener("error", () => setTimeout(startStream, 1500));
setInterval(() => {
  if (lastState && lastState.camera_ok && Date.now() - lastFrameAt > 4000) startStream();
}, 2000);

// ---------- staff settings: hold the title for 3 seconds (or press C on a keyboard) ----------
let holdTimer = null, staffTimer = null;
const title = $("#title");
title.addEventListener("pointerdown", () => { holdTimer = setTimeout(openStaff, 3000); });
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) title.addEventListener(ev, () => clearTimeout(holdTimer));

function openStaff() {
  if (!staff.open) staff.showModal();
  if (lastState) paintStaff(lastState);
  clearTimeout(staffTimer);
  staffTimer = setTimeout(() => staff.close(), 120000);
}
function paintStaff(s) {
  $("#cal-status").textContent = s.calibrating ? s.calibration_message
    : s.calibrated ? "Calibrated. The 5-foot range is set." : "Not calibrated yet. Every face is blurred until it is.";
  $("#cal-start").disabled = s.calibrating;
  $("#fact-switch").textContent = s.last_switch_ms == null ? "none yet" : `${s.last_switch_ms} ms from tap to new picture`;
  $("#fact-fps").textContent = `${s.fps} frames per second`;
  $("#fact-screen").textContent = `${innerWidth} by ${innerHeight} pixels`;
  const sizes = [...looks.children].map((b) => b.getBoundingClientRect());
  const w = Math.min(...sizes.map((r) => r.width)), h = Math.min(...sizes.map((r) => r.height));
  $("#fact-card").textContent = `${Math.round(w)} by ${Math.round(h)} pixels`;
  $("#fact-filters").textContent = s.filters.map((f) => f.available ? f.name : `${f.name} (off: ${f.error})`).join(", ");
}
$("#cal-start").addEventListener("click", async () => {
  await api("/api/calibrate", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  poll(true);
});
$("#staff-close").addEventListener("click", () => staff.close());

// Keyboard (if one is plugged in): 1 to 4 pick a filter, C opens staff settings
document.addEventListener("keydown", (e) => {
  if (staff.open) return;
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= filters.length && filters[n - 1].available) choose(filters[n - 1].id);
  if (e.key === "c" || e.key === "C") openStaff();
});

// A kiosk: no long-press menus or dragging
document.addEventListener("contextmenu", (e) => e.preventDefault());
document.addEventListener("dragstart", (e) => e.preventDefault());

let resizeTimer = null;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(fitScreen, 150); });
// Falling snow is off unless the address ends in ?snow=1 (turn it on only if the Pi stays fast;
// the staff panel shows the speed)
if (new URLSearchParams(location.search).get("snow") === "1") document.body.classList.add("falling-snow");
fitScreen();
startStream();
poll();
