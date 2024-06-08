def brute_force_duplicate(array):
    for i in range(len(array)-1):
        for j in range(i+1, len(array)):
            if array[i] == array[j]:
                return True
    return False

array = [1, 2, 43, 44, 2, 1, 5]
print(brute_force_duplicate(array))



