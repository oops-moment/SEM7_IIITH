import bisect
class Solution:
    def solve(self,citations:list[int])->int:
        citations = sorted(citations)
        for i in range(len(citations), -1, -1):
            index = bisect.bisect_right(citations, i-1)
            if len(citations) - index >= i:
                return i
        
        return 0  

if __name__=='__main__':
    solution=Solution()
    citations=[3,0,6,1,5]
    print(solution.solve(citations))