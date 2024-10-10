class Solution :
    def convert(self,n:int,input:str)->str:
        store = [""]*n
        j=0
        direction=-1

        for char in input:
            store[j]+=char
            if(j==0 or j==n-1):
                direction*=-1
            
            if(direction==-1):
                j=j-1
            else :
                j=j+1
        
        return "".join(store)

if __name__ == '__main__' :
    input_string=input("Enter String ")
    numRows = int(input("Enter number of row"))
    solutionClass=Solution()
    print(solutionClass.convert(numRows,input_string))
