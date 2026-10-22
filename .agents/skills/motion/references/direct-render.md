# Direct browser rendering

Use this route for editable HTML/CSS/SVG/Canvas animations. Share one deterministic animation function between preview and export. The reusable Node script is [../scripts/render-code-video.mjs](../scripts/render-code-video.mjs).

The page must expose:

```js
window.__motion = {
  ready: true, // only after fonts/images/resources resolve
  duration: 24, width: 1920, height: 1080,
  render(t) { /* draw the complete frame at t seconds */ },
  snapshot() { return canvas.toDataURL('image/png'); }
};
```

`?export=1` should disable interactive playback and hide controls. `?format=vertical` must actually compose the portrait scene and expose its dimensions. The script accepts landscape/vertical/square labels but cannot implement a missing page layout. Ensure a fully composed opening frame or intentional first-frame poster. Font loading failures, clipping and DOM overlays must be caught visually; Canvas alone does not report semantic text overflow.

Dependencies: Node.js, playwright-core, an installed Chrome/Edge or Playwright Chromium, and FFmpeg with libx264. Prefer existing/bundled runtime packages. Resolve the Codex bundle using load_workspace_dependencies when available, then pass its Node executable and node_modules directory. `--playwright-root` is a module directory; omit it when playwright-core resolves normally. Browser defaults to `msedge`; use `--browser chrome` or `--browser chromium` as appropriate. No MCP video service is used.

```text
node scripts/render-code-video.mjs --project ./film --mode preview --port 5273
node scripts/render-code-video.mjs --project ./film --mode qa --out ./qa --times 0,3.4,3.5,7.49,7.5,12,16,20.5,23.98
node scripts/render-code-video.mjs --project ./film --mode video --out ./film.mp4 --audio ./film/assets/music.wav --ffmpeg /path/to/ffmpeg --fps 60
```

Pass `--format vertical` for the portrait page, `--playwright-root /path/to/node_modules` for a bundled package, and `--overwrite` only when deliberately replacing an output. No audio argument produces a silent MP4. Supply a full-duration track, because the script does not extend a short input track. New work should use a new output folder.

QA saves key-frame PNGs and compares ordered rendering with a direct seek. It checks runtime/asset errors and flat frames, not creative quality. Inspect the PNGs, verify all claims/readability, and watch the complete render. The script checks FFmpeg exit status and performs full decoding; its JSON records planned dimensions/frame count, so verify those again from the actual encoded stream rather than treating the JSON as an independent probe.

Inspect with ffprobe when available:

```text
ffprobe -v error -count_frames -show_entries stream=codec_type,width,height,r_frame_rate,nb_read_frames:format=duration -of json film.mp4
ffmpeg -v error -i film.mp4 -f null -
```

Use FFmpeg's input metadata and decoded frame progress if ffprobe is unavailable. Pick a settled poster frame and retain editable source plus the soundtrack/asset credits.
