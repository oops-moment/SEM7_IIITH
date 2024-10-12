class solution :
    def removeElement(self, nums1:list[int],val:int)->list[int]:
        index1=0
        index2=0
        while(index2 < len(nums1)):
            if(nums1[index2]==val):
                index2=index2+1
            else:
                nums1[index1]=nums1[index2]
                index1=index1+1
                index2=index2+1
        return nums1
    
if __name__=='__main__':
    solutionclass=solution()
    nums = [3,2,2,3]
    val = 3
    print(solutionclass.removeElement(nums,val))