# Fish Audio narration

The user explicitly selected Fish Audio on October 8, 2026 and wants to choose the voice. Use its connected MCP for requested narration. Do not require Fish for an instrumental or silent video. Do not infer that Fish's TTS tools generate a soundtrack or SFX.

## Voice selection and synthesis

1. Discover currently callable Fish Audio tools. Use `search_voices` with `self=true` for the authorized workspace's voices, or language/query filters for suitable public voices. Offer a small relevant choice with sample URLs. Ask which voice the user likes when no choice is supplied; accept a voice ID, name, or delivery description. Preserve an already provided voice choice.
2. Resolve the selected ID with `get_voice`. Do not invent IDs from names, and do not silently substitute another voice. Use `get_credit_balance` to confirm the connected team, available credits and per-call byte limit. Voice selection does not authorize creating a clone or changing the connected team.
3. Write a concise narration script that complements the visuals. Inline delivery tags such as `[confident]` or `[warm]` are supported by the current `text_to_speech` tool; use them sparingly and validate against the live schema. Stay within its UTF-8 byte budget, counting tags. The inspected tool charges by text bytes, with some cross-team professional voices costing double; read current tool/account details rather than assuming a free call.
4. Call `text_to_speech` only for requested speech, with the actual voice ID. An ad may use short scene clips to control pacing or a continuous track followed by alignment. Avoid generating many speculative takes. The inspected tool allows one identical retry within five minutes of a failed/timed-out call; follow its current retry terms.
5. Preserve the returned audio URL/view link and voice/script information in the project's credits or manifest. Download the generated audio only when needed for local video composition. If the service returns further required steps, follow them. Do not fabricate a completed local mix when only a remote audio result exists.

## Place the voice in the film

Measure actual clip duration and silence. Align scene entrances/emphasis to the spoken words, or place each short clip inside its storyboard slot. When adapting the kit's continuous narration, regenerate both alignment and audio features and update every changed word/line anchor.

Keep a dry voice stem, background music/SFX and the final mixed track. Duck music during speech, preserve breathing room, avoid clipped peaks, and verify the voice is intelligible throughout. Prefer retiming motion or revising a phrase over chopping off speech. Export and decode the narrated MP4, then check complete playback and audio/visual sync. An optional voice variant can preserve a previously delivered instrumental version.

When Fish Audio is explicitly selected, use this Motion audio path with direct code, the kit, or a verified composition backend. Brag's installed full `--voice` workflow prescribes Kokoro; borrowing Brag's repository/storyboard guidance does not require using that provider. Respect the user's Fish choice rather than invoking a mismatched Brag narration flag.

No audio provider is required for Motion itself. If Fish is disconnected or has insufficient credits, state the concrete limitation and complete the storyboard/source and independent render work. Voice, model/team access and available credits must be checked live.
