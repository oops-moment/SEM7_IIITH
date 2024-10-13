import sys
class Solution(object):
    def recurse(self,buy,prices,index,dp_array):
        if index == len(prices):
            if buy == 0 :
                return -10000
            else :
                return 0
        if dp_array[buy][index]!=-1:
            return dp_array[buy][index]

        if buy :
            choose = -prices[index]+self.recurse(0,prices,index + 1,dp_array)
            notchoose=self.recurse(1,prices,index+1,dp_array)
            dp_array[buy][index]=max(choose,notchoose)
            return max(choose,notchoose)

        else :
            choose =prices[index] + self.recurse(1,prices, index+1,dp_array)
            notchoose=self.recurse(0,prices,index+1,dp_array)
            dp_array[buy][index]=max(choose,notchoose)
            return max(choose,notchoose)
        
        return 0
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        buy=1
        dp_array=[[-1]*len(prices) for _ in range(2)]
        return self.recurse(buy,prices,0,dp_array)

if __name__=='__main__':
    solutionClass=Solution()
    numberdays=int(input("Number of days3 "))
    prices=[]
    for i in range(numberdays):
        prices.append(int(input("Enter the price ")))
    solution= solutionClass.maxProfit(prices)
    print(solution)