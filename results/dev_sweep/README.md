# E2 dev eval (dev_a: 64 GSM8K-train + 64 MATH-train held-out practice problems, 1 sample, 32k, think35)

Paired vs base on identical problems. shift/sel/lp_* use Astra's u; real* use a real-loop mean-diff u from our own traces (no Astra input).
Selected among ~14 candidates on this set: confirm on final_practice before claiming differences.

| Edit | Changed weights | GSM8K acc | MATH acc | Pooled acc | Gain (SE) | Rescues / losses | Cut % | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| astra_p50c | 130,188 | 70.3 | 48.4 | 59.4 | +48.4 (4.7) | 64 / 2 | 12.5 | 16,502 |
| base | 0 | 10.9 | 10.9 | 10.9 | +0.0 (0.0) | 0 / 0 | 89.1 | 29,868 |
| lp_sel | 1,284,815 | 71.9 | 54.7 | 63.3 | +52.3 (4.4) | 67 / 0 | 21.9 | 16,774 |
| lp_shift | 772,252 | 67.2 | 46.9 | 57.0 | +46.1 (4.7) | 61 / 2 | 24.2 | 14,214 |
| random_ctrl | 130,285 | 9.4 | 10.9 | 10.2 | -0.8 (2.1) | 3 / 4 | 92.2 | 30,661 |
| realsel_k512 | 1,025,282 | 56.2 | 42.2 | 49.2 | +38.3 (4.3) | 49 / 0 | 49.2 | 20,971 |
| realsel_k64 | 128,710 | 57.8 | 45.3 | 51.6 | +40.6 (4.5) | 53 / 1 | 45.3 | 20,042 |
| realshift_k64 | 123,121 | 65.6 | 46.9 | 56.2 | +45.3 (4.6) | 59 / 1 | 35.9 | 17,079 |
| realshift_k8 | 16,271 | 62.5 | 46.9 | 54.7 | +43.8 (4.5) | 57 / 1 | 36.7 | 17,425 |
| recipe_nosearch | 130,152 | 53.1 | 35.9 | 44.5 | +33.6 (4.2) | 43 / 0 | 48.4 | 20,719 |
| sel_k512 | 1,029,493 | 68.8 | 48.4 | 58.6 | +47.7 (4.6) | 62 / 1 | 22.7 | 16,253 |
| sel_k64 | 129,458 | 68.8 | 48.4 | 58.6 | +47.7 (4.6) | 62 / 1 | 21.1 | 15,475 |
| shift_k128 | 256,623 | 60.9 | 46.9 | 53.9 | +43.0 (4.7) | 57 / 2 | 25.0 | 14,716 |
| shift_k32 | 64,927 | 70.3 | 46.9 | 58.6 | +47.7 (4.6) | 62 / 1 | 25.0 | 14,278 |
| shift_k8 | 16,291 | 71.9 | 53.1 | 62.5 | +51.6 (4.6) | 67 / 1 | 21.9 | 14,501 |
