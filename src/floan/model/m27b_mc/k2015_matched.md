# W3b matched comparison — anchor Dec2014 (k=2015), char scheme

Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on the **identical 150,000-loan subsample**. Comparators priced by exact matrix composition; GRU / transformer by Monte-Carlo path simulation. Same population, pools, realized reference and cashflow engine — the gap is the learner.

| H | Empirical | Logit | FF current-state | Ensemble ×8 | GRU (MC) | Transformer (MC) |
|---|---|---|---|---|---|---|
| 1 | 0.430 (-0.430) | 0.186 (-0.172) | 0.273 (-0.271) | 0.263 (-0.262) | 0.357 (-0.357) | 0.398 (-0.398) |
| 3 | 0.220 (-0.017) | 0.197 (+0.158) | 0.114 (-0.033) | 0.125 (+0.005) | 0.088 (-0.058) | 0.120 (-0.104) |
| 6 | 0.184 (-0.058) | 0.148 (+0.123) | 0.140 (-0.136) | 0.109 (-0.083) | 0.118 (-0.111) | 0.160 (-0.147) |
| 12 | 0.206 (-0.200) | 0.116 (+0.110) | 0.126 (-0.109) | 0.118 (-0.093) | 0.163 (-0.157) | 0.212 (-0.206) |
