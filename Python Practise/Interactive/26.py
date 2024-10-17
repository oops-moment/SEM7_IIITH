class Solution:
    def solve(self,nums:list[int])->int:
        index1=1
        for i in range(len(nums)):
            if i > 0:
                if nums[i] != nums[i-1]:
                    nums[index1]=nums[i]
                    index1 = index1 + 1
        
        return index1

if __name__=='__main__':
    solutionClass=Solution()
    nums=[1,1,2]
    print(solutionClass.solve(nums))