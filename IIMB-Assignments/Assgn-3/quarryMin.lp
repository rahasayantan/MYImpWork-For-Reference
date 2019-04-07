\* quarryMin *\
Minimize
Objective: 1000 carryFwd_0 + 1000 carryFwd_1 + 1000 carryFwd_2
 + 1000 carryFwd_3 + 1000 carryFwd_4 + 1000 carryFwd_5 + 1000 carryFwd_6
 + 215000 q1Stones_1 + 215000 q1Stones_2 + 215000 q1Stones_3
 + 207500 q1Stones_4 + 207500 q1Stones_5 + 207500 q1Stones_6
 + 240000 q2Stones_1 + 240000 q2Stones_2 + 240000 q2Stones_3
 + 232500 q2Stones_4 + 232500 q2Stones_5 + 232500 q2Stones_6
Subject To
Demand_Constraints_6: q1Stones_1 + q1Stones_2 + q1Stones_3 + q1Stones_4
 + q1Stones_5 + q1Stones_6 + q2Stones_1 + q2Stones_2 + q2Stones_3 + q2Stones_4
 + q2Stones_5 + q2Stones_6 >= 7000
No_Carry_at_first_period: carryFwd_0 = 0
No_Carry_at_last_period: carryFwd_6 = 0
Production_Constraints_Q1_1: q1Stones_1 <= 800
Production_Constraints_Q1_2: q1Stones_2 <= 800
Production_Constraints_Q1_3: q1Stones_3 <= 800
Production_Constraints_Q1_4: q1Stones_4 <= 800
Production_Constraints_Q1_5: q1Stones_5 <= 800
Production_Constraints_Q1_6: q1Stones_6 <= 800
Production_Constraints_Q2_1: q2Stones_1 <= 1400
Production_Constraints_Q2_2: q2Stones_2 <= 1400
Production_Constraints_Q2_3: q2Stones_3 <= 1400
Production_Constraints_Q2_4: q2Stones_4 <= 1400
Production_Constraints_Q2_5: q2Stones_5 <= 1400
Production_Constraints_Q2_6: q2Stones_6 <= 1400
Stone_Storage_Limit_1: carryFwd_1 <= 1200
Stone_Storage_Limit_2: carryFwd_2 <= 1200
Stone_Storage_Limit_3: carryFwd_3 <= 1200
Stone_Storage_Limit_4: carryFwd_4 <= 1200
Stone_Storage_Limit_5: carryFwd_5 <= 1200
Stone_Storage_Limit_6: carryFwd_6 <= 1200
Stopped_Production_1: q2Stones_4 = 0
Stopped_Production_2: q2Stones_5 = 0
_C1: carryFwd_0 - carryFwd_1 + q1Stones_1 + q2Stones_1 - unMetDemand_0
 + unMetDemand_1 <= 700
_C2: carryFwd_1 - carryFwd_2 + q1Stones_2 + q2Stones_2 - unMetDemand_1
 + unMetDemand_2 <= 700
_C3: carryFwd_2 - carryFwd_3 + q1Stones_3 + q2Stones_3 - unMetDemand_2
 + unMetDemand_3 <= 1000
_C4: carryFwd_3 - carryFwd_4 + q1Stones_4 + q2Stones_4 - unMetDemand_3
 + unMetDemand_4 <= 1200
_C5: carryFwd_4 - carryFwd_5 + q1Stones_5 + q2Stones_5 - unMetDemand_4
 + unMetDemand_5 <= 2000
_C6: carryFwd_5 - carryFwd_6 + q1Stones_6 + q2Stones_6 - unMetDemand_5
 + unMetDemand_6 <= 1400
x: unMetDemand_0 = 0
End
