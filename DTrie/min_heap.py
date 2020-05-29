class MinHeap(object):
    def __init__(self, arr, max_size=10):
        self.arr = arr
        self.arr_pos_map = {arr[i][1]: i for i in range(len(arr))}
        self.max_size = max_size

    def heapify_index(self, index):
        j = index
        n = len(self.arr)

        while True:
            if 2 * j + 2 < n and self.arr[j][0] > min(self.arr[2 * j + 1][0], self.arr[2 * j + 2][0]):
                p = min(self.arr[2 * j + 1][0], self.arr[2 * j + 2][0])

                if p == self.arr[2 * j + 1][0]:
                    temp = self.arr[j]
                    self.arr[j] = self.arr[2 * j + 1]
                    self.arr[2 * j + 1] = temp

                    self.arr_pos_map[self.arr[j][1]] = j
                    self.arr_pos_map[self.arr[2 * j + 1][1]] = 2 * j + 1
                    j = 2 * j + 1
                else:
                    temp = self.arr[j]
                    self.arr[j] = self.arr[2 * j + 2]
                    self.arr[2 * j + 2] = temp

                    self.arr_pos_map[self.arr[j][1]] = j
                    self.arr_pos_map[self.arr[2 * j + 2][1]] = 2 * j + 2
                    j = 2 * j + 2

            elif 2 * j + 1 < n and self.arr[j][0] > self.arr[2 * j + 1][0]:
                temp = self.arr[j]
                self.arr[j] = self.arr[2 * j + 1]
                self.arr[2 * j + 1] = temp

                self.arr_pos_map[self.arr[j][1]] = j
                self.arr_pos_map[self.arr[2 * j + 1][1]] = 2 * j + 1
                j = 2 * j + 1

            else:
                break

    def heapify(self):
        n = len(self.arr)

        for i in range(int(n / 2) - 1, -1, -1):
            self.heapify_index(i)

    def push(self, elem):
        self.arr.append(elem)
        n = len(self.arr)
        j = n - 1

        self.arr_pos_map[elem[1]] = j

        parent = int(j / 2) - 1 if j % 2 == 0 else int(j / 2)

        while parent >= 0 and self.arr[parent] > self.arr[j]:
            temp = self.arr[parent]
            self.arr[parent] = self.arr[j]
            self.arr[j] = temp

            self.arr_pos_map[self.arr[parent][1]] = parent
            self.arr_pos_map[self.arr[j][1]] = j

            j = parent
            parent = int(j / 2) - 1 if j % 2 == 0 else int(j / 2)

    def pop(self):
        if len(self.arr) == 0:
            raise Exception('No element in heap to pop')

        if len(self.arr) == 1:
            return self.arr.pop()

        x, y = self.arr[0], self.arr.pop()

        self.arr[0] = y
        self.arr_pos_map[y[1]] = 0

        self.heapify_index(0)

        return x

    def update_key(self, key, value):
        if key in self.arr_pos_map:
            j = self.arr_pos_map[key]
            self.arr[j] = (value, key)
            self.heapify_index(j)

    def pop_n_push(self, elem):
        if len(self.arr) < self.max_size:
            self.push(elem)
        else:
            if elem[0] > self.arr[0][0]:
                self.arr[0] = elem
                self.arr_pos_map[elem[1]] = 0

                self.heapify_index(0)