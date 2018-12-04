class Solution(object):
    def get_digits(self, D, N):
        num_digits = len(N)
        
        if num_digits == 0:
            return 0
        
        n, full_sum = len(D), 0

        for i in range(num_digits):
            full_sum += n**i
        
        partial_sum = full_sum - n**(num_digits-1)
        
        out, first_digit = 0, N[0]
        
        for x in D:
            if int(x) < int(first_digit):
                out += full_sum
            elif int(x) == int(first_digit):
                out += 1 + self.get_digits(D, N[1:])
            else:
                out += partial_sum
        
        return out
    
    def atMostNGivenDigitSet(self, D, N):
        out = self.get_digits(D, str(N))
        return out
