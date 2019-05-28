class AllOne(object):

    def __init__(self):
        self.map1 = dict()
        self.map2 = dict()
        self.head_cnt, self.tail_cnt = None, None
        
        
    def delete(self, key, cnt):
        if cnt in self.map2:
            self.map2[cnt]["keys"].remove(key)
            p, n = self.map2[cnt]["prev"], self.map2[cnt]["next"]
            flag = True

            if len(self.map2[cnt]["keys"]) == 0:
                self.map2.pop(cnt)

                if p in self.map2:
                    self.map2[p]["next"] = n
                else:
                    self.head_cnt = n

                if n in self.map2:
                    self.map2[n]["prev"] = p
                else:
                    self.tail_cnt = p
                flag = False
            return p, n, flag
        return None, None, None
    
    
    def add(self, key, cnt, p, n):
        if cnt not in self.map2:
            self.map2[cnt] = {"keys":set([key]), "prev":None, "next":None}
        else:
            self.map2[cnt]["keys"].add(key)
            
        if p in self.map2:
            if p < cnt:
                self.map2[p]["next"] = cnt
                self.map2[cnt]["prev"] = p
        else:
            self.head_cnt = cnt
            
        if n in self.map2:
            if n > cnt:
                self.map2[n]["prev"] = cnt
                self.map2[cnt]["next"] = n
        else:
            self.tail_cnt = cnt
        

    def inc(self, key):
        if key not in self.map1:
            self.map1[key] = 1
            self.add(key, 1, None, self.head_cnt)
            
        else:
            p, n, flag = self.delete(key, self.map1[key])
            if flag:
                p = self.map1[key]
                
            self.map1[key] += 1
            self.add(key, self.map1[key], p, n)
        

    def dec(self, key):
        if key in self.map1:
            p, n, flag = self.delete(key, self.map1[key])
            if flag:
                n = self.map1[key]
            self.map1[key] -= 1
            if self.map1[key] == 0:
                self.map1.pop(key)
            else:
                self.add(key, self.map1[key], p, n)
        

    def getMaxKey(self):
        if self.tail_cnt in self.map2:
            return next(iter(self.map2[self.tail_cnt]["keys"]))
        return ""
        

    def getMinKey(self):
        if self.head_cnt in self.map2:
            return next(iter(self.map2[self.head_cnt]["keys"]))
        return ""
