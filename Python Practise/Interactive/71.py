class Solution:
    def simplify_path(self,input_path : str)->str :
        stack=[]
        directories=input_path.split('/')
        for dir in directories:
            if dir=="." or not dir :
                continue
            if dir=="..":
                if stack:
                    stack.pop()
            else :
                stack.append(dir)
        
        return "/"+"/".join(stack)


if __name__=='__main__':
    solutionClass=Solution()
    input_path=str(input("Enter the input  "))
    print(solutionClass.simplify_path(input_path))