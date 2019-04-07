\* fuelMin *\
Minimize
Objective: 11 d1_0 - 10 d1_1 + 8.8 d2_0 - 8 d2_1 + 6.6 d3_0 - 6 d3_1
Subject To
Demand_Constraint_for_R_Fuel: R_1 + R_2 + R_3 + R_4 + R_5 <= 4000
Demand_Constraint_for_S_Fuel: S_1 + S_2 + S_3 + S_4 + S_5 <= 2900
Demand_Constraint_for_U_Fuel: U_1 + U_2 + U_3 + U_4 + U_5 <= 4900
Integer_Constraint_for_One_Type_of_Fuel_in_Container_1: Yr__1 + Ys__1 + Yu__1
 <= 1
Integer_Constraint_for_One_Type_of_Fuel_in_Container_2: Yr__2 + Ys__2 + Yu__2
 <= 1
Integer_Constraint_for_One_Type_of_Fuel_in_Container_3: Yr__3 + Ys__3 + Yu__3
 <= 1
Integer_Constraint_for_One_Type_of_Fuel_in_Container_4: Yr__4 + Ys__4 + Yu__4
 <= 1
Integer_Constraint_for_One_Type_of_Fuel_in_Container_5: Yr__5 + Ys__5 + Yu__5
 <= 1
Maximum_Capacity_of_R_Fuel_if_Container_1_has_R: R_1 - 2700 Yr__1 <= 0
Maximum_Capacity_of_R_Fuel_if_Container_2_has_R: R_2 - 2800 Yr__2 <= 0
Maximum_Capacity_of_R_Fuel_if_Container_3_has_R: R_3 - 1100 Yr__3 <= 0
Maximum_Capacity_of_R_Fuel_if_Container_4_has_R: R_4 - 1800 Yr__4 <= 0
Maximum_Capacity_of_R_Fuel_if_Container_5_has_R: R_5 - 3400 Yr__5 <= 0
Maximum_Capacity_of_S_Fuel_if_Container_1_has_S: S_1 - 2700 Ys__1 <= 0
Maximum_Capacity_of_S_Fuel_if_Container_2_has_S: S_2 - 2800 Ys__2 <= 0
Maximum_Capacity_of_S_Fuel_if_Container_3_has_S: S_3 - 1100 Ys__3 <= 0
Maximum_Capacity_of_S_Fuel_if_Container_4_has_S: S_4 - 1800 Ys__4 <= 0
Maximum_Capacity_of_S_Fuel_if_Container_5_has_S: S_5 - 3400 Ys__5 <= 0
Maximum_Capacity_of_U_Fuel_if_Container_1_has_U: U_1 - 2700 Yu__1 <= 0
Maximum_Capacity_of_U_Fuel_if_Container_2_has_U: U_2 - 2800 Yu__2 <= 0
Maximum_Capacity_of_U_Fuel_if_Container_3_has_U: U_3 - 1100 Yu__3 <= 0
Maximum_Capacity_of_U_Fuel_if_Container_4_has_U: U_4 - 1800 Yu__4 <= 0
Maximum_Capacity_of_U_Fuel_if_Container_5_has_U: U_5 - 3400 Yu__5 <= 0
Maximum_Shortfall_(500)_Constraint_for_R_Fuel: R_1 + R_2 + R_3 + R_4 + R_5
 >= 3500
Maximum_Shortfall_(500)_Constraint_for_S_Fuel: S_1 + S_2 + S_3 + S_4 + S_5
 >= 2400
Maximum_Shortfall_(500)_Constraint_for_U_Fuel: U_1 + U_2 + U_3 + U_4 + U_5
 >= 4400
_C1: S_1 + S_2 + S_3 + S_4 + S_5 + d1_0 - d1_1 = 2650
_C2: R_1 + R_2 + R_3 + R_4 + R_5 + d2_0 - d2_1 = 3750
_C3: U_1 + U_2 + U_3 + U_4 + U_5 + d3_0 - d3_1 = 4650
Binaries
Yr__1
Yr__2
Yr__3
Yr__4
Yr__5
Ys__1
Ys__2
Ys__3
Ys__4
Ys__5
Yu__1
Yu__2
Yu__3
Yu__4
Yu__5
End
