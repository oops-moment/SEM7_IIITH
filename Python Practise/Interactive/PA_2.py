class AccountBalanceManager:

    def manage(self, input_commands: list[str]) -> list[str]:
        final_answer = {}  # Should be a list instead of a dictionary based on the function return type.
        init_commands = []
        other_commands = []
        userDetails = {}

        # Extract all the init commands
        for command in input_commands:
            if command.startswith("INIT"):
                init_commands.append(command)
            else:
                other_commands.append(command)
        
        # Initialise the user details from the init command
        for command in init_commands:
            command = command.split(",")
            name = command[1]
            balance = int(command[2])
            banks = command[3:]
            userDetails[name] = {
                'balance': balance,
                'banks': banks
            }
        
        # Sort all other commands in order of the transaction time 
        other_commands = sorted(other_commands, key=lambda x: int(x.split(",")[1]))

        for command in other_commands:
            original_command = command
            command = command.split(",")
            
            if command[0] == "GET":
                name = command[2]
                if name in userDetails.keys():
                    final_answer[original_command] = userDetails[name]['balance']
                else: 
                    final_answer[original_command] = "FAILURE"

            elif command[0] == "POST":
                # Check if both are in userDetails, then initiate the transfer
                sender = command[2]
                receiver = command[3]
                amount = int(command[4])

                if sender in userDetails.keys() and receiver in userDetails.keys():
                    if amount > userDetails[sender]['balance']:
                        final_answer[original_command] = "FAILURE"
                    else:
                        userDetails[sender]['balance'] -= amount
                        userDetails[receiver]['balance'] += amount
                        final_answer[original_command] = "SUCCESS"
                    continue  # Continue is unnecessary; the loop will proceed anyway.
                
                if sender not in userDetails.keys() and receiver not in userDetails.keys():
                    final_answer[original_command] = "FAILURE"
                    continue

                if sender not in userDetails.keys():
                    # Mismatched quotes here will cause a syntax error.
                    # Incorrectly checks if the sender is not in the user's banks (should be checking receiver).
                    if sender not in userDetails[receiver]['banks']:  # Remove the extra quote
                        final_answer[original_command] = "FAILURE"  # Spelling error "Failre"
                    else:
                        userDetails[receiver]['balance'] += amount  # Missing response after this operation.
                
                if receiver not in userDetails.keys():
                    # The comparison should check 'receiver', not 'sender'.
                    if receiver not in userDetails[sender]['banks']:
                        final_answer[original_command] = "FAILURE"  # Typo: "FAI;LUREA"
                    else:
                        userDetails[sender]['balance'] -= amount

            elif command[0] == "ADD_BANK":
                # If the user does not exist, and it is the new bank, we would keep the balance as 0
                name = command[2]
                bank = command[3]
                if name in userDetails.keys():
                    # Typo: `userDetails[name]['bank']` should be `['banks']`
                    banks = userDetails[name]['banks']  
                    banks.append(bank)
                    userDetails[name]['banks'] = banks
                else:
                    userDetails[name] = {
                        'balance': 0,  # Assignment operator is incorrect: should be a colon, not `=`
                        'banks': [bank]  # 'banks' should be a list, not a string.
                    }
                final_answer[original_command] = "SUCCESS"

            else:  # This is the case of removing the bank
                name = command[2]
                bank = command[3]
                if name in userDetails.keys():
                    # Same typo: should be 'banks', not 'bank'.
                    banks = userDetails[name]['banks']
                    if bank in banks:
                        banks.remove(bank)
                        userDetails[name]['banks'] = banks
                final_answer[original_command] = "SUCCESS"

        return final_answer  # Should return a list, not a dictionary.
    
if __name__ == '__main__':

    N = int(input("Enter number of commands: "))
    input_commands = []

    for _ in range(N):
        input_command = input("Enter the Command: ")
        input_commands.append(input_command)
    
    accountManager = AccountBalanceManager()
    final_answer = accountManager.manage(input_commands)

    for id, result in final_answer.items():
        print("ID: ", id, " Value: ", result)  # This is assuming `final_answer` is a dictionary, but the function should return a list.