import re

class Solution:
    def solve(self, input_names: list[str]) -> list[str]:
        company2id = {}
        final_answer = []
        
        remove_strings = ["the", "an", "a"]
        suffixes = ["inc.", "corp.", "llc", "l.l.c.", "llc."]

        for input_name in input_names:
            id, company_name = input_name.split('|')
            
            # 1. Convert to lower case and remove unwanted characters
            company_name = company_name.lower()
            company_name = company_name.replace('&', ' ').replace(',', ' ')
            company_name = company_name.strip() 
            company_name = ' '.join(company_name.split()) 

            # 2. Remove leading "The", "An", "A"
            company_name_parts = company_name.split()
            if company_name_parts[0] in remove_strings:
                company_name_parts = company_name_parts[1:]
            company_name = ' '.join(company_name_parts)
            
            # 3. Remove "and" unless it's the first word
            if " and " in company_name and not company_name.startswith("and "):
                company_name = company_name.replace(" and ", ' ')
            company_name = ' '.join(company_name.split())  # Re-normalize spaces

            for suffix in suffixes:
                if company_name.endswith(suffix):
                    company_name = company_name[:-(len(suffix))].strip()

            # 5. Check if the transformed name is available
            if not company_name or company_name in company2id:
                final_answer.append(f"{id} | Name Not Available")
            else:
                final_answer.append(f"{id} | Name Available")
                company2id[company_name] = 1
        
        return final_answer

if __name__ == '__main__':
    solution = Solution()
    n = int(input("Enter total number of names: "))
    input_strings = []
    for _ in range(n):
        input_string = input()
        input_strings.append(input_string)
    
    result = solution.solve(input_strings)
    for res in result:
        print(res)
