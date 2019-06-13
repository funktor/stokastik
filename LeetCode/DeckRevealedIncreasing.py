class Solution(object):
    def deckRevealedIncreasing(self, deck):
        values = [0]*len(deck)
        deck = sorted(deck)
        values[0] = deck[0]
        
        if len(deck) == 1:
            return values
        
        i, j, k = 1, 1, 1
        while True:
            if j == 2 and values[i] == 0:
                values[i] = deck[k]
                k += 1
                j = 1
            elif values[i] == 0:
                j += 1
                
            if k == len(deck):
                return values
            
            i = (i+1) % len(deck)
