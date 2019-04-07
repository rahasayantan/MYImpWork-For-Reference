\* hireMin *\
Minimize
Objective: 36000 analyst_1 + 34800 hiredEmpl_1 + 29100 hiredEmpl_2
 + 12000 hiredEmpl_3 + 6000 hiredEmpl_4 + 20000 isReqHire_1
 + 20000 isReqHire_2 + 8000 transfrEmpl_1 + 8000 transfrEmpl_2
 + 8000 transfrEmpl_3 + 8000 transfrEmpl_4 + 8000 transfrEmpl_5
 + 8000 transfrEmpl_6
Subject To
Demand_Dec: analyst_1 + 0.95 hiredEmpl_1 + 0.95 hiredEmpl_2
 + 0.8 transfrEmpl_4 >= 65
Demand_Feb: analyst_1 + 0.95 hiredEmpl_1 + 0.95 hiredEmpl_2 + hiredEmpl_3
 + hiredEmpl_4 + 0.8 transfrEmpl_6 >= 90
Demand_Jan: analyst_1 + 0.95 hiredEmpl_1 + 0.95 hiredEmpl_2 + hiredEmpl_3
 + 0.8 transfrEmpl_5 >= 80
Demand_Nov: analyst_1 + 0.95 hiredEmpl_1 + hiredEmpl_2 + 0.8 transfrEmpl_3
 >= 90
Demand_Oct: analyst_1 + hiredEmpl_1 + hiredEmpl_2 + 0.8 transfrEmpl_2 >= 105
Demand_Sept: analyst_1 + hiredEmpl_1 + 0.8 transfrEmpl_1 >= 110
Initial_Number_of_confirmed_Analyst_as_of_Sept01: analyst_1 = 79
Transferred_Employee_20%_Constraint: 1.2 analyst_1 + 1.16 hiredEmpl_1
 + 0.97 hiredEmpl_2 + 0.4 hiredEmpl_3 + 0.2 hiredEmpl_4 - transfrEmpl_1
 - transfrEmpl_2 - transfrEmpl_3 - transfrEmpl_4 - transfrEmpl_5
 - transfrEmpl_6 >= 0
_C1: hiredEmpl_1 - 50 isReqHire_1 <= 0
_C2: hiredEmpl_2 - 50 isReqHire_1 <= 0
_C3: hiredEmpl_3 - 50 isReqHire_2 <= 0
_C4: hiredEmpl_4 - 50 isReqHire_2 <= 0
Bounds
0 <= hiredEmpl_1
0 <= hiredEmpl_2
0 <= hiredEmpl_3
0 <= hiredEmpl_4
0 <= transfrEmpl_1
0 <= transfrEmpl_2
0 <= transfrEmpl_3
0 <= transfrEmpl_4
0 <= transfrEmpl_5
0 <= transfrEmpl_6
Generals
hiredEmpl_1
hiredEmpl_2
hiredEmpl_3
hiredEmpl_4
transfrEmpl_1
transfrEmpl_2
transfrEmpl_3
transfrEmpl_4
transfrEmpl_5
transfrEmpl_6
Binaries
isReqHire_1
isReqHire_2
End
