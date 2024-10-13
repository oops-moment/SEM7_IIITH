class Solution:
    def solve(self,nums):
        i = 0
        store = 0
        for i in range(len(nums)):
            if(store < 0):
                return False
            else :
                store=store-1
                store=max(store,nums[i])
                if(i!=len(nums)-1 and store <=0):
                    return False
        
        return True

if __name__ == '__main__':
    solution=Solution()
    nums=[2,3,1,1,4]
    print(solution.solve(nums))