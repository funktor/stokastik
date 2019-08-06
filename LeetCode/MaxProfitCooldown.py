class Solution(object):
    def maxProfit(self, prices):
        if len(prices) <= 1:
            return 0
        
        profit = [0]*len(prices)
        max_profit, max_profit_index = -1, -1
        
        for i in range(1, len(prices)):
            for j in range(i):
                if j+2 < i:
                    profit[i] = max(profit[i], prices[i]-prices[j], profit[j] + prices[i]-prices[j+2])
                else:
                    profit[i] = max(profit[i], prices[i]-prices[j], profit[j])
        
        return profit[-1]
        
