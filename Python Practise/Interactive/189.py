class Solution:
    def solve(self,nums:list[int],k:int):
        k=k%len(nums)

        nums[0:]=nums[0:][::-1]
        nums[0:k]=nums[0:k][::-1]
        nums[k:]=nums[k:][::-1]

if __name__=='__main__':
    solution=Solution()
    nums = [1,2,3,4,5,6,7]
    k = 3
    solution.solve(nums,k)
    print(nums)
