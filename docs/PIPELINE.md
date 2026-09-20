# SignSpeak Pipeline Documentation

## Dataset Inspection (Phase 0 & 2)

- **Total videos:** 76 videos across 10 sign classes
- **Pose detection:** 95% - 100% across all classes (overall ~98%)
- **Hand detection:** Average 58% - 81% across full clips (hands at rest during start/end of clips)
- **Sign classes:** `friend` (8), `goodbye` (2), `hello` (5), `help` (6), `name` (9), `no` (12), `please` (8), `sorry` (7), `thank_you` (6), `yes` (13)
- **Flagged videos:** 49 videos have <70% hand detection when averaged over the full clip duration because the signer's hands rest at the start/end of the clip. During active signing segments, hands are tracked consistently.
- **Orientation:** All videos are upright. Unmirrored raw frames are passed to MediaPipe everywhere.

## Segmentation and Windowing (Phase 1 & 2)

Signing activity is detected when:
1. Pose is present.
2. At least one hand is detected.
3. The detected hand wrist is above the hip line (`wrist_y < hip_y` in normalized coordinates where y increases downwards).
4. Temporal smoothing: gaps < 0.3s merged, segments < 0.35s dropped, padded 0.2s on each side.

### Segment Duration Percentiles

| Label | Segments | p50 (s) | p75 (s) | p90 (s) | p95 (s) | Max (s) |
|---|---|---|---|---|---|---|
| friend | 8 | 2.07 | 2.33 | 2.50 | 2.67 | 2.84 |
| goodbye | 2 | 1.69 | 1.76 | 1.81 | 1.82 | 1.84 |
| hello | 5 | 1.84 | 1.84 | 2.82 | 3.14 | 3.47 |
| help | 6 | 1.82 | 1.89 | 2.05 | 2.13 | 2.20 |
| name | 9 | 1.71 | 1.90 | 2.00 | 2.13 | 2.27 |
| no | 12 | 1.96 | 2.75 | 3.15 | 3.35 | 3.59 |
| please | 7 | 2.03 | 2.46 | 2.62 | 2.66 | 2.70 |
| sorry | 7 | 1.90 | 2.70 | 3.14 | 3.43 | 3.72 |
| thank_you | 6 | 1.74 | 1.96 | 2.25 | 2.38 | 2.50 |
| yes | 13 | 1.44 | 1.83 | 2.47 | 2.67 | 2.76 |
| **OVERALL** | **75** | **1.84** | **2.20** | **2.74** | **3.12** | **3.72** |

### Window Configuration
- **Calculated WINDOW_S:** `clip(p90 + 0.4, 1.5, 3.0) = clip(2.74 + 0.4, 1.5, 3.0) = 3.00s`.
- **Segments exceeding 1.5 * WINDOW_S (4.50s):** 0 of 75 segments. No splitting required.
- **`INFER_STRIDE_MS`:** 100 ms.
- **`SEQ_LEN`:** 32 frames.
- **Feature dimension:** 258 features per frame (Pose: 33x4 = 132, LH: 21x3 = 63, RH: 21x3 = 63).
