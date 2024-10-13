class Solution:
    def solve(self,nums:list[int])->list:
        N=len(nums)
        left_products=[1]*N
        right_products=[1]*N
        final_answer=[1]*N

        for i in range(1,N):
            left_products[i]=left_products[i-1]*nums[i-1]
        
        for i in range(N-2,-1,-1):
            right_products[i]=right_products[i+1]*nums[i+1]
        

        for i in range(N):
            final_answer[i]=left_products[i]*right_products[i]

        return final_answer

if __name__=='__main__':
    solutionclass=Solution()
    N = int(input("Enter number of element"))
    input_array=[]
    for _ in range(N):
        input_array.append(int(input("Enter num ")))
    print(solutionclass.solve(input_array))
    
