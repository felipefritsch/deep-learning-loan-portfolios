# W3b matched comparison — anchor Dec2024 (k=2025), char scheme

Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on the **identical 150,000-loan subsample**. Comparators priced by exact matrix composition; GRU / transformer by Monte-Carlo path simulation. Same population, pools, realized reference and cashflow engine — the gap is the learner.

| H | Empirical | Logit | FF current-state | Ensemble ×8 | GRU (MC) | Transformer (MC) |
|---|---|---|---|---|---|---|
| 1 | 0.887 (-0.887) | 0.310 (-0.082) | 0.197 (-0.122) | 0.208 (-0.128) | 0.176 (-0.025) | 0.166 (-0.048) |
| 3 | 0.708 (-0.708) | 0.280 (-0.016) | 0.152 (-0.049) | 0.162 (-0.067) | 0.174 (+0.036) | 0.176 (+0.062) |
| 6 | 0.665 (-0.665) | 0.262 (+0.049) | 0.134 (-0.077) | 0.146 (-0.101) | 0.133 (-0.009) | 0.114 (+0.004) |
| 12 | 0.562 (-0.556) | 0.197 (+0.194) | 0.144 (+0.126) | 0.134 (+0.115) | 0.220 (+0.220) | 0.227 (+0.227) |
