\* gpPortfolioBlend *\
Minimize
Objective: d1_1 + d2_1 + d3_0
Subject To
_C1: - d1_0 + d1_1 + 4.2 x1 + 3.5 x2 + 2.8 x3 + 2.5 x4 + 0.2 x5 >= 100
_C2: - d2_0 + d2_1 + 4.2 x1 + 3.5 x2 >= 20
_C3: d3_0 - d3_1 - 0.2 x5 >= -5
_C4: 120000 x1 + 85000 x2 + 70000 x3 + 60000 x4 + 25000 x5 <= 2000000
Bounds
0 <= x1
0 <= x2
0 <= x3
0 <= x4
0 <= x5
Generals
x1
x2
x3
x4
x5
End
