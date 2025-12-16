from cs336_alignment.drgrpo_grader import grade_answer_sympy

prediction = """
To calculate the total cost, we need to consider the pricing structure. Since every second glass costs 60% of $5, which is $3, we can divide the total number of glasses by 2 to find the number of discounted glasses. Kylar wants to buy 16 glasses, so there will be 8 glasses at the discounted price. The cost for the discounted glasses is 8 * $3 = $24. The remaining 8 glasses cost $5 each, so their total cost is 8 * $5 = $40. The total cost is the sum of the discounted and full-price glasses, which is $24 + $40 = $64. </think> <answer> $64 </answer>
"""

ground_truth = "'The discount price of one glass is 60/100 * 5 = $<<60/100*5=3>>3.\nIf every second glass is cheaper, that means Kylar is going to buy 16 / 2 = <<16/2=8>>8 cheaper glasses.\nSo for the cheaper glasses, Kylar is going to pay 8 * 3 = $<<8*3=24>>24.\nAnd for the regular-priced glasses, Kylar will pay 8 * 5 = $<<8*5=40>>40.\nSo in total Kylar needs to pay 24 + 40 = $<<24+40=64>>64 for the glasses he wants to buy.\n#### 64'"

score = grade_answer_sympy(prediction, ground_truth)
