from typing import List

class Solution:
    def __init__(self) -> None:
        self.cache = {}

    def dp(self, input_strings: List[str], counter: List[List[int]], m: int, n: int, idx: int) -> int:
        if m < 0 or n < 0:
            return -100000
        if idx == len(input_strings):
            return 0
        
        key = (m, n, idx)
        if key in self.cache:
            return self.cache[key]

        self.cache[key] = max(
            self.dp(input_strings, counter, m, n, idx + 1),  
            1 + self.dp(input_strings, counter, m - counter[idx][0], n - counter[idx][1], idx + 1)
        )
        
        return self.cache[key]
              
    def solve(self, m: int, n: int, input_strings: List[str]) -> int:
        counter = [[s.count("0"), s.count("1")] for s in input_strings]
        return self.dp(input_strings, counter, m, n, 0)

if __name__ == '__main__':
    num_strings = int(input("Enter the number of strings: "))
    
    input_strings = []
    for _ in range(num_strings):
        string = input(f"Enter string {_ + 1}: ")
        input_strings.append(string)
    
    m = int(input("Enter the maximum number of zeroes: "))
    n = int(input("Enter the maximum number of ones: "))

    solution = Solution()
    print(solution.solve(m, n, input_strings))
