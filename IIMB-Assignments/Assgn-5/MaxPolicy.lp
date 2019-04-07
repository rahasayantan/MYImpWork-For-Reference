\* MaxPolicy *\
Maximize
Objective: demand_1 + demand_2 + demand_3 + demand_4
Subject To
Initial_Demand: demand_0 = 5.84471771198
Initial_Discount: discount_0 = 0.579
Initial_Inventory: inventory_0 = 18.8627302777
Initial_Price: price_0 = 606
_C1: demand_1 + inventory_0 - inventory_1 >= 0
_C10: - 0.56468 demand_2 + demand_3 + 3.19977 discount_2 - 3.89091 discount_3
 <= 1.90154190378
_C11: - discount_2 + discount_3 >= 0
_C12: demand_3 - inventory_2 <= 0
_C13: demand_4 + inventory_3 - inventory_4 >= 0
_C14: - 0.56468 demand_3 + demand_4 + 3.19977 discount_3 - 3.89091 discount_4
 <= 1.88279873899
_C15: - discount_3 + discount_4 >= 0
_C16: demand_4 - inventory_3 <= 0
_C17: demand_1 + demand_2 + demand_3 + demand_4 - inventory_0 <= 0
_C2: - 0.56468 demand_0 + demand_1 + 3.19977 discount_0 - 3.89091 discount_1
 <= 1.93931734438
_C3: - discount_0 + discount_1 >= 0
_C4: demand_1 - inventory_0 <= 0
_C5: demand_2 + inventory_1 - inventory_2 >= 0
_C6: - 0.56468 demand_1 + demand_2 + 3.19977 discount_1 - 3.89091 discount_2
 <= 1.92038094342
_C7: - discount_1 + discount_2 >= 0
_C8: demand_2 - inventory_1 <= 0
_C9: demand_3 + inventory_2 - inventory_3 >= 0
Bounds
0.579 <= discount_0 <= 0.6
0.579 <= discount_1 <= 0.6
0.579 <= discount_2 <= 0.6
0.579 <= discount_3 <= 0.6
0.579 <= discount_4 <= 0.6
End
