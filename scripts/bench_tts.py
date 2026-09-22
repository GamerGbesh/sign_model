"""Benchmark script for Amegbe TTS engine.

Measures:
1. Model load time.
2. Cold synthesis latency for 1-word and 6-word sentences.
3. Cached synthesis latency.
4. Real-Time Factor (RTF = synthesis_time / audio_duration).
"""
import shutil
import tempfile
import time
from pathlib import Path

from asl.tts import PiperStableTwiBackend, TwiTTS


def benchmark():
    print("=" * 60)
    print("AMEGBE TTS BENCHMARK")
    print("=" * 60)

    # 1. Model load time
    t0 = time.perf_counter()
    backend = PiperStableTwiBackend()
    backend._ensure_loaded()
    load_time = time.perf_counter() - t0
    print(f"Model Load Time:          {load_time:.3f} s")

    # Temp cache directory for benchmark
    temp_cache = Path(tempfile.mkdtemp(prefix="amegbe_bench_"))
    try:
        tts = TwiTTS(backend=backend, cache_dir=temp_cache, start_worker=False)

        # 2. Cold 1-word
        w_text = "nsuo"
        t0 = time.perf_counter()
        res_w = tts.speak(w_text)
        w_latency = time.perf_counter() - t0
        w_dur = res_w.duration_s if res_w else 0.0
        w_rtf = (w_latency / w_dur) if w_dur > 0 else 0.0

        print("\n[Cold Single Word: 'nsuo']")
        print(f"  Synthesis Latency:      {w_latency * 1000:.2f} ms")
        print(f"  Audio Duration:         {w_dur:.2f} s")
        print(f"  Real-Time Factor (RTF): {w_rtf:.4f} ({1/w_rtf:.1f}x real-time)")

        # 3. Cold 6-word sentence
        s_text = "Akwaaba, wo ho te sɛn paa?"
        t0 = time.perf_counter()
        res_s = tts.speak(s_text)
        s_latency = time.perf_counter() - t0
        s_dur = res_s.duration_s if res_s else 0.0
        s_rtf = (s_latency / s_dur) if s_dur > 0 else 0.0

        print("\n[Cold Sentence (6 words): 'Akwaaba, wo ho te sɛn paa?']")
        print(f"  Synthesis Latency:      {s_latency * 1000:.2f} ms")
        print(f"  Audio Duration:         {s_dur:.2f} s")
        print(f"  Real-Time Factor (RTF): {s_rtf:.4f} ({1/s_rtf:.1f}x real-time)")

        # 4. Cached latency (100 iterations)
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            tts.speak(s_text)
        cached_avg_ms = ((time.perf_counter() - t0) / iters) * 1000

        print("\n[Cached Audio Lookup (100 runs)]")
        print(f"  Average Latency:        {cached_avg_ms:.3f} ms")

        # Summary assertions
        assert w_rtf < 0.25, f"1-word RTF too high: {w_rtf}"
        assert s_rtf < 0.15, f"Sentence RTF too high: {s_rtf}"
        assert cached_avg_ms < 5.0, f"Cached lookup too slow: {cached_avg_ms} ms"
        print("\nAll benchmark targets MET (RTF < 0.15, Cache < 5ms).")

    finally:
        shutil.rmtree(temp_cache, ignore_errors=True)


if __name__ == "__main__":
    benchmark()
