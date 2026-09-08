"""Does the rest of the design hold at ten people? Arithmetic, not opinion.

Four capacity questions, none of which involve clustering:

  1. ASR — how many requests and how many seconds of audio per minute of
     meeting, finals and partials, against the one shared MERaLiON endpoint.
  2. The auto-prompt — how many "who is this?" cards fire, and when.
  3. The transcript — how many segments, and what that does to the browser.
  4. The minutes — whether an hour of ten people fits the token budget.
"""
from __future__ import annotations

import numpy as np

from meeting import turns

WORDS_PER_SECOND = 2.5          # ~150 wpm conversational
TOKENS_PER_WORD = 1.35          # English, with speaker labels and timestamps


def asr_load(n: int, minutes: float, partials: bool,
             interval_early=1.5, interval_late=3.0, widen_after=6.0,
             seeds=range(6)) -> dict:
    """Requests per minute and audio-seconds per minute, at the endpoint."""
    reqs, audio, speech = [], [], []
    for seed in seeds:
        segs = turns(n, minutes, seed)
        r = len(segs)                      # one final per segment
        a = sum(s.seconds for s in segs)
        speech.append(a)
        if partials:
            for s in segs:
                # Ticks while the segment is open, widening after `widen_after`.
                t, ticks = interval_early, []
                while t < s.seconds:
                    ticks.append(t)
                    t += interval_early if t < widen_after else interval_late
                r += len(ticks)
                a += sum(ticks)            # each partial re-sends from the start
        reqs.append(r)
        audio.append(a)
    return {
        "requests_per_min": float(np.mean(reqs)) / minutes,
        "audio_sec_per_min": float(np.mean(audio)) / minutes,
        "speech_sec_per_min": float(np.mean(speech)) / minutes,
        "realtime_factor": float(np.mean(audio)) / (minutes * 60),
    }


def prompt_storm(n: int, minutes: float, min_seconds=6.0, min_segments=2,
                 seeds=range(6)) -> dict:
    """When does each speaker become promptable, and how many pile up?"""
    firsts = []
    for seed in seeds:
        segs = turns(n, minutes, seed)
        secs = {i: 0.0 for i in range(n)}
        cnt = {i: 0 for i in range(n)}
        t = 0.0
        ready: list[float] = []
        done = set()
        for s in segs:
            t += s.seconds + 0.6
            secs[s.speaker] += s.seconds
            cnt[s.speaker] += 1
            if s.speaker not in done and secs[s.speaker] >= min_seconds \
                    and cnt[s.speaker] >= min_segments:
                done.add(s.speaker)
                ready.append(t)
        ready.sort()
        firsts.append(ready)
    k = min(len(f) for f in firsts)
    med = [float(np.median([f[i] for f in firsts])) for i in range(k)]
    within_2min = float(np.mean([sum(1 for x in f if x <= 120) for f in firsts]))
    return {"ready_times": med, "all_ready_at": med[-1] if med else 0.0,
            "ready_within_2min": within_2min, "n_promptable": k}


def transcript_volume(n: int, minutes: float, seeds=range(6)) -> dict:
    segs = [turns(n, minutes, s) for s in seeds]
    counts = [len(x) for x in segs]
    words = [sum(t.seconds for t in x) * WORDS_PER_SECOND for x in segs]
    return {
        "segments": float(np.mean(counts)),
        "segments_per_min": float(np.mean(counts)) / minutes,
        "words": float(np.mean(words)),
        "tokens": float(np.mean(words)) * TOKENS_PER_WORD,
    }


def minutes_budget(tokens: float, chunk=4000, overlap=200, out_per_chunk=400,
                   reduce_out=2000, ctx=32768) -> dict:
    step = chunk - overlap
    chunks = max(1, int(np.ceil((tokens - overlap) / step)))
    reduce_in = chunks * out_per_chunk
    return {
        "chunks": chunks,
        "map_calls": chunks,
        "reduce_input_tokens": reduce_in,
        "reduce_total_tokens": reduce_in + reduce_out,
        "fits_context": (reduce_in + reduce_out) < ctx,
        "single_pass_would_need": tokens + reduce_out,
        "single_pass_fits": (tokens + reduce_out) < ctx,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("ASR LOAD — one shared MERaLiON endpoint, 60-minute meeting")
    print("=" * 72)
    print(f"{'people':>6} {'mode':>10} {'req/min':>9} {'audio s/min':>12} "
          f"{'x realtime':>11}")
    for n in (4, 10, 15):
        for partials in (False, True):
            r = asr_load(n, 60.0, partials)
            print(f"{n:>6} {'partials' if partials else 'finals':>10} "
                  f"{r['requests_per_min']:>9.1f} {r['audio_sec_per_min']:>12.1f} "
                  f"{r['realtime_factor']:>10.2f}x")

    print()
    print("=" * 72)
    print("AUTO-PROMPT — how many cards, how fast")
    print("=" * 72)
    for n in (4, 10, 15):
        p = prompt_storm(n, 60.0)
        t = [f"{x/60:.1f}" for x in p["ready_times"][:12]]
        print(f"{n:>3} people: {p['n_promptable']} promptable, "
              f"{p['ready_within_2min']:.1f} ready within 2 min, "
              f"last at {p['all_ready_at']/60:.1f} min")
        print(f"          ready at (min): {' '.join(t)}")

    print()
    print("=" * 72)
    print("TRANSCRIPT VOLUME + MINUTES TOKEN BUDGET — 60 minutes")
    print("=" * 72)
    for n in (4, 10, 15):
        v = transcript_volume(n, 60.0)
        b = minutes_budget(v["tokens"])
        print(f"{n:>3} people: {v['segments']:>5.0f} segments "
              f"({v['segments_per_min']:.1f}/min), {v['words']:>6.0f} words, "
              f"{v['tokens']:>6.0f} tokens")
        print(f"          map-reduce: {b['chunks']} chunks -> reduce sees "
              f"{b['reduce_total_tokens']} tokens (fits: {b['fits_context']}); "
              f"single pass would need {b['single_pass_would_need']:.0f} "
              f"(fits: {b['single_pass_fits']})")


# --------------------------------------------------------------- capacity

#: voicebot measures MERaLiON on vLLM at ~600 ms for a short utterance. Taking
#: that utterance as ~3 s of audio gives the endpoint's throughput on one
#: sequential request stream. vLLM's continuous batching does better than this
#: under concurrency, so treating it as the ceiling is the conservative read.
ASR_XRT_PER_STREAM = 3.0 / 0.6          # ~5x realtime


def capacity(meetings: int, partials: bool, live_call: bool,
             n=10, minutes=60.0) -> dict:
    """How much of the shared recogniser a given load actually asks for."""
    per = asr_load(n, minutes, partials)["realtime_factor"]
    used = meetings * per
    # A live voicebot call is one utterance at a time, ~3 s of audio every
    # ~8 s of call: well under 1x realtime, but it is latency-critical.
    call = 3.0 / 8.0 if live_call else 0.0
    return {
        "per_meeting_xrt": per,
        "meetings_xrt": used,
        "call_xrt": call,
        "total_xrt": used + call,
        "capacity_xrt": ASR_XRT_PER_STREAM,
        "headroom": ASR_XRT_PER_STREAM - (used + call),
        "utilisation": (used + call) / ASR_XRT_PER_STREAM,
    }


def capacity_report() -> None:
    print("=" * 78)
    print("SHARED-ENDPOINT CAPACITY — 10-person meetings, conservative 5x "
          "realtime ceiling")
    print("=" * 78)
    print(f"{'meetings':>8} {'partials':>9} {'live call':>10} {'demand':>9} "
          f"{'headroom':>9} {'util':>7}  verdict")
    for meetings in (1, 2, 3):
        for partials in (True, False):
            for call in (True, False):
                c = capacity(meetings, partials, call)
                verdict = ("OK" if c["utilisation"] < 0.7 else
                           "TIGHT" if c["utilisation"] < 1.0 else "OVER")
                print(f"{meetings:>8} {str(partials):>9} {str(call):>10} "
                      f"{c['total_xrt']:>8.2f}x {c['headroom']:>8.2f}x "
                      f"{c['utilisation']*100:>6.0f}%  {verdict}")
