class Solution :
    def solve(self,num1:str,num2:str)->str:
        num1=int(num1)
        num2 = int(num2)
        return str(num1*num2)

if __name__ == '__main__':
    solution_class= Solution()
    num1=str(input("Enter num1 "))
    num2=str(input("Enter num2 "))
    print(solution_class.solve(num1,num2)) 
