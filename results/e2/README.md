# E2 dev eval (dev_a: 64 GSM8K-train + 64 MATH-train held-out practice problems, 1 sample, 32k, think35)

Paired vs base on identical problems. LP edits here use Astra's u (the 'given the diagnosis' setting).

| Edit | Changed weights | GSM8K acc | MATH acc | Pooled acc | Gain (SE) | Rescues / losses | Cut % | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| astra_p50c | 130,188 | 70.3 | 48.4 | 59.4 | +48.4 (4.7) | 64 / 2 | 12.5 | 16,502 |
| base | 0 | 10.9 | 10.9 | 10.9 | +0.0 (0.0) | 0 / 0 | 89.1 | 29,868 |
| lp_sel | 1,284,815 | 71.9 | 54.7 | 63.3 | +52.3 (4.4) | 67 / 0 | 21.9 | 16,774 |
| lp_shift | 772,252 | 67.2 | 46.9 | 57.0 | +46.1 (4.7) | 61 / 2 | 24.2 | 14,214 |
| random_ctrl | 130,285 | 9.4 | 10.9 | 10.2 | -0.8 (2.1) | 3 / 4 | 92.2 | 30,661 |
| recipe_nosearch | 130,152 | 53.1 | 35.9 | 44.5 | +33.6 (4.2) | 43 / 0 | 48.4 | 20,719 |
