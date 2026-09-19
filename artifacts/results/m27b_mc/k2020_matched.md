# W3b matched comparison — anchor Dec2019 (k=2020), char scheme

Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on the **identical 150,000-loan subsample**. The ladder runs memoryless (empirical / logit / FF-current / ensemble, priced by exact matrix composition) → engineered history (FF + history, Monte-Carlo) → learned memory (GRU / transformer, Monte-Carlo). Same population, pools, realized reference and cashflow engine — the gap is the learner.

| H | Empirical | Logit | FF current-state | Ensemble ×8 | FF + history (MC) | GRU (MC) | Transformer (MC) |
|---|---|---|---|---|---|---|---|
| 1 | 0.250 (-0.046) | 0.148 (-0.024) | 0.101 (+0.075) | 0.123 (+0.107) | 0.180 (+0.162) | 0.104 (-0.067) | 0.094 (-0.073) |
| 3 | 0.344 (+0.293) | 0.250 (+0.227) | 0.285 (+0.273) | 0.313 (+0.305) | 0.345 (+0.337) | 0.206 (+0.182) | 0.188 (+0.164) |
| 6 | 0.505 (+0.505) | 0.441 (+0.441) | 0.479 (+0.479) | 0.511 (+0.511) | 0.514 (+0.514) | 0.358 (+0.358) | 0.365 (+0.365) |
| 12 | 0.553 (+0.553) | 0.560 (+0.560) | 0.565 (+0.565) | 0.582 (+0.582) | 0.620 (+0.620) | 0.485 (+0.485) | 0.493 (+0.493) |
