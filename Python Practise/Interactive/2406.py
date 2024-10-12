class Solution:
    def solve(self,input_intervals:list[list[int]])->int:
        maxm=-1
        for interval in input_intervals:
            maxm=max(maxm,interval[1])
        
        store_array=[0]*int(maxm+2)
        for interval in input_intervals:
            store_array[interval[0]]+=1
            store_array[interval[1]+1]-=1
        
        maxm=1
        
        for i in range(len(store_array)):
            if i>0:
                store_array[i]=store_array[i]+store_array[i-1]
                maxm=max(maxm,store_array[i])
        
        return maxm
        

if __name__=='__main__':
    solution=Solution()
    input_intervals=[[5,10],[6,8],[1,5],[2,3],[1,10]]
    input_interals2=[[1,3],[5,6],[8,10],[11,13]]
    print(solution.solve(input_intervals))
    print(solution.solve(input_interals2))