class LFUCache(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.map = dict()
        self.m = dict()
        self.head_cnt, self.tail_cnt = None, None
        
    def delete(self, key, cnt):
        p, n = self.m[cnt][key]
        
        if p in self.m[cnt]:
            self.m[cnt][p][1] = n
        else:
            self.m[cnt]["head"] = n

        if n in self.m[cnt]:
            self.m[cnt][n][0] = p
        else:
            self.m[cnt]["tail"] = p
            
        self.m[cnt].pop(key)
        
        if self.m[cnt]["head"] == None or self.m[cnt]["tail"] == None:
            p_cnt, n_cnt = self.m[cnt]["prev"], self.m[cnt]["next"]
            self.m.pop(cnt)
            return p_cnt, n_cnt
        
        return cnt, self.m[cnt]["next"]
    
    def add(self, key, cnt):
        if cnt not in self.m:
            self.m[cnt] = {key:[None, None], "head":key, "tail":key, "prev":None, "next":None}
                
        else:
            self.m[cnt][key] = [self.m[cnt]["tail"], None]
            self.m[cnt][self.m[cnt]["tail"]][1] = key
            self.m[cnt]["tail"] = key
            
        
    def adjust(self, key, value=None):
        out, cnt = self.map[key]
        val = out if value is None else value
        
        p_cnt, n_cnt = self.delete(key, cnt)
        self.add(key, cnt + 1)
        
        self.m[cnt + 1]["prev"] = p_cnt
        if p_cnt in self.m:
            self.m[p_cnt]["next"] = cnt + 1
        else:
            self.head_cnt = cnt + 1
            
        if n_cnt > cnt + 1:
            self.m[cnt + 1]["next"] = n_cnt
            
        if n_cnt in self.m:
            if n_cnt > cnt + 1:
                self.m[n_cnt]["prev"] = cnt + 1
        else:
            self.tail_cnt = cnt + 1
            
        self.map[key] = [val, cnt + 1]
            
        return val
    

    def get(self, key):
        if key in self.map:
            out = self.adjust(key)
            return out
        return -1
        

    def put(self, key, value):
        if self.capacity > 0:
            if len(self.map) == self.capacity and key not in self.map:
                k = self.m[self.head_cnt]["head"]
                self.map.pop(k)
                self.m[self.head_cnt]["head"] = self.m[self.head_cnt][k][1]
                if self.m[self.head_cnt]["head"] is not None:
                    self.m[self.head_cnt][self.m[self.head_cnt]["head"]][0] = None
                self.m[self.head_cnt].pop(k)

                if self.m[self.head_cnt]["head"] == None or self.m[self.head_cnt]["tail"] == None:
                    n_cnt = self.m[self.head_cnt]["next"]
                    if n_cnt in self.m:
                        self.m[n_cnt]["prev"] = None
                    self.m.pop(self.head_cnt)
                    self.head_cnt = n_cnt

            if len(self.map) == 0:
                self.map[key] = [value, 1]
                self.m[1] = {key:[None, None], "head":key, "tail":key, "prev":None, "next":None}
                self.head_cnt, self.tail_cnt = 1, 1

            elif key not in self.map:
                self.map[key] = [value, 1]

                if 1 not in self.m:
                    self.m[1] = {key:[None, None], "head":key, "tail":key, "prev":None, "next":None}
                else:
                    self.m[1][key] = [self.m[1]["tail"], None]
                    self.m[1][self.m[1]["tail"]][1] = key
                    self.m[1]["tail"] = key

                if self.head_cnt > 1:
                    self.m[1]["next"] = self.head_cnt
                    self.m[self.head_cnt]["prev"] = 1
                    self.head_cnt = 1

            else:
                self.adjust(key, value)
