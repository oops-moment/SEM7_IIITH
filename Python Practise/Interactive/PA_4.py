
# str = stripe.com/payments/checkout/customer.john.doe
# minor_parts = 2
# after Part 1 compression
#=>
# s4e.c1m/pos/c6t/cr.j2n.dle
# after Part 2 compression
#=>
# s4e.c1m/p6s/c6t/c6r.j5e
class Solution:
    def solve(self,input_string:str)->str:
        final_answer=[]
        input_string=input_string.split("/")
        for major_part in input_string:
            major_part=major_part.split(".")
            minor_parts=[]
            for minor_part in major_part:
                start_letter=minor_part[0]
                end_letter=minor_part[-1]
                count=len(minor_part)-2
                value=f"{start_letter}{count}{end_letter}"
                minor_parts.append(value)
            final_answer.append(".".join(minor_parts))
    
        return "/".join(final_answer)


if __name__=='__main__':
    solutionClass=Solution()
    input_string=input("Enter the string ")
    print(solutionClass.solve(input_string))