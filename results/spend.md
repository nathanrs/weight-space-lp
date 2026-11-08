# Cloud spend ledger (cap: $100)

| Date | Cluster | GPU | Hours | $ | Purpose |
|---|---|---|---|---|---|
| 2026-11-06/07 | wslp (Lambda us-west-3) | 1× H100 PCIe @ $3.29/h | ~6.1 | ~$20 | Jobs 1–6: E0 harness gate, E1 fit traces, E2 dev evals (14 edits), confirmation |
| 2026-11-07 | wslp (Lambda us-west-3) | 1× H100 PCIe @ $3.29/h | ~12.2 | ~$40 | Jobs 1–2: E3 finalists (GSM8K test, MATH-500 ×4, blind confirmation), IFEval, blind δ sweep |
| 2026-11-07 | wslp (Lambda us-east-1) | 1× A100 SXM4 @ $1.99/h (no H100 capacity) | ~1.5 | ~$3 | E4 steering baseline (dev sweep + MATH-500) |

**Total: ~$63 of $100.**
