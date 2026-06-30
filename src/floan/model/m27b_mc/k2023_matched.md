# W3b matched comparison — anchor Dec2022 (k=2023), char scheme

Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on the **identical 150,000-loan subsample**. Comparators priced by exact matrix composition; GRU / transformer by Monte-Carlo path simulation. Same population, pools, realized reference and cashflow engine — the gap is the learner.

| H | Empirical | Logit | FF current-state | Ensemble ×8 | GRU (MC) | Transformer (MC) |
|---|---|---|---|---|---|---|
| 1 | 1.178 (-1.178) | 0.298 (-0.231) | 0.161 (-0.025) | 0.122 (+0.044) | 0.151 (-0.117) | 0.238 (-0.225) |
| 3 | 0.854 (-0.854) | 0.290 (+0.007) | 0.271 (+0.206) | 0.294 (+0.270) | 0.165 (+0.071) | 0.164 (+0.035) |
| 6 | 0.846 (-0.846) | 0.299 (+0.007) | 0.226 (+0.101) | 0.236 (+0.155) | 0.177 (-0.055) | 0.145 (-0.059) |
| 12 | 0.984 (-0.984) | 0.238 (-0.068) | 0.206 (-0.047) | 0.169 (+0.067) | 0.156 (-0.034) | 0.185 (-0.146) |
