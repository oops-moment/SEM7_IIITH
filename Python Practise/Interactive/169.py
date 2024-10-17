class Solution:
    def solve(self,nums:list[int])->int:
        element=-1
        frequency=0

        for num in nums :
            if element == -1:
                element=num
                frequency=frequency +1
            else :
                if element == num :
                    frequency = frequency + 1
                else :
                    frequency = frequency - 1
                    if frequency == 0:
                        element=-1
        
        return element


if __name__=='__main__':
    solutionClass=Solution()
    nums=[1,2,3,3,3,5,5,5,5]
    print(solutionClass.solve(nums))