class Solution(object):
    def maxProfit(self, prices, fee):
        if len(prices) <= 1:
            return 0
        
        profit = [0]*len(prices)
        curr_max_util = - prices[0] - fee
        
        for i in range(1, len(prices)):
            profit[i] = max(profit[i], profit[i-1], prices[i] + curr_max_util)
            curr_max_util = max(curr_max_util, profit[i-1] - prices[i] - fee)
        
        return profit[-1]
