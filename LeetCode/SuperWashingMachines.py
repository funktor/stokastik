class Solution(object):
    def findMinMoves(self, machines):
        sum = 0
        for num in machines:
            sum += num

        if sum % len(machines) != 0:
            return -1

        target = int(sum / len(machines))

        max_c, left_req, right_pass = 0, 0, 0

        for idx in range(len(machines)):

            if machines[idx] < target:
                if right_pass + machines[idx] < target:
                    req = target - machines[idx] - right_pass
                    left_req += req
                    right_pass = 0
                    max_c = max(max_c, left_req, right_pass, left_req + right_pass)
                else:
                    right_pass += machines[idx] - target
                    max_c = max(max_c, left_req, right_pass, left_req + right_pass)

            elif machines[idx] > target:
                balance = machines[idx] - target

                if left_req >= balance:
                    left_req -= balance
                    max_c = max(max_c, left_req, right_pass, left_req + right_pass)
                else:
                    right_pass += balance - left_req
                    max_c = max(max_c, left_req, right_pass, left_req + right_pass)
                    left_req = 0

        return max_c


sol = Solution()
print(sol.findMinMoves([9,1,8,8,9]))