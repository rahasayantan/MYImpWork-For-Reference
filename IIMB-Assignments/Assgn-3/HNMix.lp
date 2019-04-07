\* HVMix *\
Maximize
Objective: 50 x_dh + 45 x_fn + 10 x_kbh + 60 x_kh
Subject To
Demand_1: x_fn <= 250
Demand_2: x_dh <= 120
Demand_3: x_kh <= 90
Demand_4: x_kbh <= 550
Non_Zero_Constraint_1: x_fn >= 0
Non_Zero_Constraint_2: x_dh >= 0
Non_Zero_Constraint_3: x_kh >= 0
Non_Zero_Constraint_4: x_kbh >= 0
Supply_1: 0.2 x_dh + 0.3 x_fn + 0.3 x_kbh + 0.4 x_kh <= 500
Supply_2: 0.5 x_dh + 0.4 x_fn + 0.8 x_kbh + 0.8 x_kh <= 450
Supply_3: 0.3 x_dh + 0.4 x_fn + 0.1 x_kbh + 0.1 x_kh <= 75
Supply_4: 0.3 x_dh + 0.2 x_fn + 0.5 x_kbh + 0.6 x_kh <= 300
Supply_5: 0.5 x_dh + 0.5 x_fn + 0.8 x_kbh + 0.4 x_kh <= 200
End
