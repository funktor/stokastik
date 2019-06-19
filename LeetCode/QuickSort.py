def find_pivot(arr):
    i, j, pivot_idx = -1, 0, len(arr)-1
    
    while j < len(arr)-1:
        if i >= 0 and arr[j] <= arr[pivot_idx]:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i += 1 
        else:
            if i == -1 and arr[j] > arr[pivot_idx]:
                i = j
        j += 1
    
    temp = arr[pivot_idx]
    arr[pivot_idx] = arr[i]
    arr[i] = temp
    
    return arr, (i+len(arr))%len(arr)

def quicksort(arr):
    if len(arr) == 1:
        return [arr[0]]
    
    if len(arr) == 0:
        return []
    
    arr, pivot = find_pivot(arr)
    
    a = quicksort(arr[:pivot])
    b = quicksort(arr[pivot+1:])
    
    return a + [arr[pivot]] + b
