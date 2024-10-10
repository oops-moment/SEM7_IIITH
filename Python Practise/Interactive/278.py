class Solution :
    def isBadVersion(self,n:int)->bool:
        return True
    
    def firstBadVersion(self, n : int)->int :
        low = 1
        high = n

        while(low < high):
            mid= low + (high-low)//2
            if self.isBadVersion(mid):
                high=mid
            else :
                low =mid+1
        return low
    
if __name__=='__main__':

    checkversion=Solution()
    n = int(input("Enter the number "))
    print(checkversion.firstBadVersion(n))