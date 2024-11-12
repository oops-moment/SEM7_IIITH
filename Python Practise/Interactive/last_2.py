from collections import defaultdict

def validate_names(company_names: list[str]) -> dict:
    availability = {}
    final_answer = {}

    for company_name in company_names:
        company_id, name = company_name.split("|")

        # Normalize the name
        name = name.lower()  # Convert to lowercase

        # Replace special characters with a space
        special_characters = [",", ".", "-", "&"]
        for char in special_characters:
            name = name.replace(char, " ")

        # Remove multiple consecutive spaces
        name = " ".join(name.split())

        # Remove leading articles
        prefixes = ["the ", "an ", "a "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]  # Trim the prefix from the start

        # Replace abbreviations with their full forms
        abbreviations = {"co": "company", "intl": "international"}
        for abbr, full_form in abbreviations.items():
            words = name.split()
            name = " ".join([full_form if word == abbr else word for word in words])

        # Final check for empty or duplicate name
        name = name.strip()  # Remove any trailing spaces
        if not name or name in availability:  # Check if empty or already taken
            final_answer[company_id] = "Name Not Available"
        else:
            final_answer[company_id] = "Name Available"
            availability[name] = -1  # Mark as unavailable

    return final_answer

if __name__ == '__main__':
    company_names = ["acct_00123|Innovate Tech Inc.", "acct_00567|INNOVATE-TECH LTD."]
    answer_dict = validate_names(company_names)

    for account_id, availability in answer_dict.items():
        print(f"{account_id} | {availability}")
