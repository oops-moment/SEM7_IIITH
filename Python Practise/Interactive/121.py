import sys
class Solution:
    def solve(self, prices:list[int])->int:
        maxm=0
        minm=sys.maxsize
        for price in prices:
            minm=min(price,minm)
            maxm=max(maxm,price-minm)
            
        return maxm

if __name__=='__main__':
    solutionClass=Solution()
    prices = [7,1,5,3,6,4]
    print(solutionClass.solve(prices))

