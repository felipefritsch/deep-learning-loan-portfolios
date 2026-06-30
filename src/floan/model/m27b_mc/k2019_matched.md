# W3b matched comparison — anchor Dec2018 (k=2019), char scheme

Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on the **identical 150,000-loan subsample**. Comparators priced by exact matrix composition; GRU / transformer by Monte-Carlo path simulation. Same population, pools, realized reference and cashflow engine — the gap is the learner.

| H | Empirical | Logit | FF current-state | Ensemble ×8 | GRU (MC) | Transformer (MC) |
|---|---|---|---|---|---|---|
| 1 | 0.763 (-0.763) | 0.244 (-0.208) | 0.190 (-0.164) | 0.211 (-0.200) | 0.209 (-0.199) | 0.182 (-0.139) |
| 3 | 0.482 (-0.482) | 0.226 (-0.032) | 0.192 (-0.009) | 0.189 (-0.046) | 0.180 (-0.059) | 0.200 (-0.004) |
| 6 | 0.230 (-0.215) | 0.274 (+0.274) | 0.190 (+0.185) | 0.160 (+0.148) | 0.148 (+0.126) | 0.178 (+0.172) |
| 12 | 0.164 (+0.020) | 0.534 (+0.534) | 0.489 (+0.489) | 0.443 (+0.443) | 0.434 (+0.434) | 0.470 (+0.470) |
