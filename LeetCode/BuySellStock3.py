class Solution(object):
    def maxProfit(self, prices):
        if len(prices) == 0:
            return 0
        
        buy, sell = 0, 0
        curr_max_sell = 0
        last_best_profit_block, last_best_profit = 0, 0
        max_profit = 0
        
        for i in reversed(range(len(prices))):
            if i == len(prices)-1 or prices[i] > prices[i+1]:
                if i == len(prices)-1:
                    buy, sell, curr_max_sell = prices[i], prices[i], prices[i]
                else:
                    profit1 = curr_max_sell-buy
                    profit2 = sell-buy
                    max_profit = max(max_profit, profit1 + last_best_profit_block, profit2 + last_best_profit)
                    
                    if prices[i] >= curr_max_sell:
                        curr_max_sell = prices[i]
                        last_best_profit_block = max(last_best_profit_block, profit1)
                        
                    w = last_best_profit
                    last_best_profit = max(profit1, profit2, last_best_profit, last_best_profit_block)
                    
                    if last_best_profit == w:
                        sell = max(sell, prices[i])
                    else:
                        sell = prices[i]
                
                    buy = prices[i]
            else:
                buy = prices[i]
                
        profit1 = curr_max_sell-buy
        profit2 = sell-buy
        max_profit = max(max_profit, profit1 + last_best_profit_block, profit2 + last_best_profit)
            
        return  max_profit 
