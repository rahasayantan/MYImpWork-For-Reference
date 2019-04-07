\* MDPPolicy2 *\
Minimize
Objective: x_s1 + x_s2
Subject To
Constraint_for_State_S1_Policy_1: 0.46 x_s1 - 0.36 x_s2 >= 20
Constraint_for_State_S1_Policy_2: 0.28 x_s1 - 0.18 x_s2 >= 30
Constraint_for_State_S2_Policy_1: - 0.27 x_s1 + 0.37 x_s2 >= -2
Constraint_for_State_S2_Policy_2: - 0.72 x_s1 + 0.82 x_s2 >= -12
End
