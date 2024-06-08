numbers = [2, 0, 1, 1, 2, 0, 1, 2]

def DNFS(numbers: list) -> list:
    length = len(numbers)
    low = 0
    high = length - 1
    mid = 0

    while mid <= high:
        if numbers[mid] == 0:
            numbers[low], numbers[mid] = numbers[mid], numbers[low]
            low+=1
            mid+=1
        elif numbers[mid] == 1:
            mid+=1
        else:
            numbers[mid], numbers[high] = numbers[high], numbers[mid]
            high-=1
    return numbers

if __name__ == "__main__":
    print(DNFS(numbers))
