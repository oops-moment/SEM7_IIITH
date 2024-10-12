class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        index1=m-1
        index2=n-1
        index3=m+n-1

        while(index1>=0 and index2>=0):
            if nums1[index1]>nums2[index2]:
                nums1[index3]=nums1[index1]
                index1-=1
                index3-=1
            else:
                nums1[index3]=nums2[index2]
                index2-=1
                index3-=1
        
        while index1>=0:
            nums1[index3]=nums1[index1]
            index3=index3-1
            index1=index1-1

        while index2>=0:
            nums1[index3]=nums2[index2]
            index3=index3-1
            index2=index2-1

        return nums1
      

if __name__=='__main__':
    solutionClass=Solution()
    nums1 = [1,2,3,0,0,0]
    m = 3
    nums2 = [2,5,6]
    n = 3
    print(solutionClass.merge(nums1,m,nums2,n))