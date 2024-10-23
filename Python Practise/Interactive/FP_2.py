class ParseData:

    def __init__(self):
        self.valid_currencies = {"EUR", "USD", "SGD", "BRL", "JPY", "ISK", "KRW"}
        self.change_currencies = {"EUR", "USD", "SGD", "BRL"}
        self.withdrawn = {}
        self.final_answer = {}
        self.output = []

    def parse_data(self, file_content):
        current_headers = None
        
        for line in file_content:
            # Check if it's a new file (ends with .csv)
            if line.endswith(".csv"):
                current_file, date = line.split("_")
                current_headers = None  # Reset headers for the new file
            else:
                if current_headers is None:
                    current_headers = line.split(",")  # Parse the headers
                else:
                    values = line.split(",")
                    data = dict(zip(current_headers, values))  # Create dict from headers and values

                    # Validate the fields: currency, amount, and evidence_due_by
                    if (data['currency'] not in self.valid_currencies or 
                        not data['amount'].isdigit() or 
                        not data['evidence_due_by'].isdigit()):
                        continue

                    disputeID = f"{current_file}{data['transaction']}"

                    # Handle "withdrawn" cases
                    if data['reason'] == "withdrawn":
                        if disputeID not in self.withdrawn or date > self.withdrawn[disputeID]:
                            self.withdrawn[disputeID] = date
                    else:
                        # Handle normal cases where the transaction is active
                        if disputeID not in self.final_answer or date < self.final_answer[disputeID]["date"]:
                            self.final_answer[disputeID] = {
                                "merchant": data["merchant"],
                                "amount": data["amount"],
                                "currency": data["currency"],
                                "evidence_due_by": data["evidence_due_by"],
                                "date": date
                            }

        # Remove disputes that were withdrawn after they were finalized
        for dispute_id, date in self.withdrawn.items():
            if dispute_id in self.final_answer and date > self.final_answer[dispute_id]["date"]:
                del self.final_answer[dispute_id]

        # Format the output
        for dispute_id, value in sorted(self.final_answer.items()):
            amount = int(value["amount"])
            
            # Format amount based on the currency
            if value["currency"] in self.change_currencies:
                amount = f"{amount / 100:.2f}"
            else:
                amount = f"{amount}.00"
            
            self.output.append(
                f"{dispute_id},{value['merchant']},{amount}{value['currency']},{value['evidence_due_by']}"
            )

        return '\n'.join(self.output)  # Return joined output for clean display

# Example usage
if __name__ == '__main__':
    file_content = [
        "VISA_20230601.csv",
        "transaction,merchant,amount,currency,evidence_due_by,reason",
        "123890132,47821,37906,USD,1686812400,fraudulent",
        "110450953,63724,12750,JPY,1686898800,duplicate",
        "JCB_20230604.csv",
        "transaction,merchant,currency,amount,evidence_due_by,reason",
        "110450953,11000,SGD,15000,1686898820,duplicate"
    ]

    parsedata = ParseData()
    print(parsedata.parse_data(file_content))
