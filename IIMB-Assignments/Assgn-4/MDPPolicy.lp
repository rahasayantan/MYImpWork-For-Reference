\* MDPPolicy *\
Minimize
Objective: x_s1 + x_s2 + x_s3 + x_s4
Subject To
Constraint_for_State_S1_Policy_1: 0.525 x_s1 - 0.2375 x_s2 - 0.1425 x_s3
 - 0.095 x_s4 >= 180
Constraint_for_State_S1_Policy_2: 0.62 x_s1 - 0.285 x_s2 - 0.1425 x_s3
 - 0.1425 x_s4 >= 160
Constraint_for_State_S1_Policy_3: 0.43 x_s1 - 0.19 x_s2 - 0.095 x_s3
 - 0.095 x_s4 >= 200
Constraint_for_State_S2_Policy_1: 0.2875 x_s2 - 0.1425 x_s3 - 0.095 x_s4
 >= 225
Constraint_for_State_S2_Policy_2: 0.2875 x_s2 - 0.095 x_s3 - 0.1425 x_s4
 >= 200
Constraint_for_State_S2_Policy_3: - 0.095 x_s1 + 0.335 x_s2 - 0.095 x_s3
 - 0.095 x_s4 >= 250
Constraint_for_State_S3_Policy_1: - 0.095 x_s2 + 0.24 x_s3 - 0.095 x_s4 >= 270
Constraint_for_State_S3_Policy_2: - 0.0475 x_s2 + 0.24 x_s3 - 0.1425 x_s4
 >= 240
Constraint_for_State_S3_Policy_3: - 0.19 x_s2 + 0.335 x_s3 - 0.095 x_s4 >= 300
Constraint_for_State_S4_Policy_1: - 0.19 x_s3 + 0.24 x_s4 >= 450
Constraint_for_State_S4_Policy_2: - 0.095 x_s3 + 0.145 x_s4 >= 400
Constraint_for_State_S4_Policy_3: - 0.285 x_s3 + 0.335 x_s4 >= 500
End
