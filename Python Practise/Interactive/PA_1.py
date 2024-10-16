import re

class Name_Validator:
    def solve(self, input_ids: list[str], input_strings: list[str]) -> list[str]:
        company_names = set()
        final_answer = {}

        for company_name, company_id in zip(input_strings, input_ids):
            # Convert to lowercase
            company_name = company_name.lower()

            # Remove special characters and replace them with spaces
            company_name = re.sub(r'[^\w\s]', ' ', company_name)

            # Convert multiple spaces into a single space
            company_name = " ".join(company_name.split()).strip()

            # Ignore leading words
            leading_words = ["the", "a", "an"]
            for leading_word in leading_words:
                if company_name.startswith(leading_word + " "):
                    company_name = company_name[len(leading_word) + 1:]

            # Replace abbreviations
            abbreviations = {'co.': "company", 'intl.': "international"}
            for key, value in abbreviations.items():
                company_name = company_name.replace(key, value)

            # Check availability
            print("company name is ",company_name)
            if not company_name or company_name in company_names:
                final_answer[company_id] = "Not Available"
            else:
                final_answer[company_id] = "Available"
                company_names.add(company_name)

        return final_answer


if __name__ == '__main__':
    name_validator = Name_Validator()
    N = int(input("Enter total number of registered startups: "))
    input_strings = []
    input_ids = []
    for _ in range(N):
        company_id, company_name = input("Enter the company ID and name (format: ID|Name): ").split("|")
        input_strings.append(company_name.strip())
        input_ids.append(company_id.strip())

    final_answer = name_validator.solve(input_ids, input_strings)
    final_answers = [f"{key} | {value}" for key, value in final_answer.items()]
    
    print("\n".join(final_answers))
