from collections import defaultdict

def account_manager(input_commands: list[str]) -> dict:
    init_commands = []
    other_commands = []

    # Dictionary to store user details: {username: {"balance": int, "banks": list[str]}}
    user_details = {}
    final_answer = {}

    # Separate INIT commands from POST and GET commands
    for command in input_commands:
        if command.split(",")[0].upper() == "INIT":
            init_commands.append(command)
        else:
            other_commands.append(command)

    # Process INIT commands to set up users with their balances and associated banks
    for command in init_commands:
        command_parts = command.split(",")
        name = command_parts[1]
        balance = int(command_parts[2])
        banks = command_parts[3:]

        user_details[name] = {
            "balance": balance,
            "banks": banks
        }

    # Sort other commands based on the timestamp
    sorted_commands = sorted(other_commands, key=lambda x: int(x.split(",")[1]))

    # Process POST and GET commands
    for command_str in sorted_commands:
        command = command_str.split(",")

        if command[0] == "POST":
            sender = command[2]
            receiver = command[3]
            amount = int(command[4])

            # Check if both sender and receiver are users
            if sender in user_details and receiver in user_details:
                # Check if sender has enough balance
                if user_details[sender]["balance"] >= amount:
                    # Transfer money
                    user_details[sender]["balance"] -= amount
                    user_details[receiver]["balance"] += amount
                    final_answer[command_str] = "SUCCESS"
                else:
                    final_answer[command_str] = "FAILURE"
            # Sender is a user, receiver is a bank
            elif sender in user_details and receiver not in user_details:
                if receiver in user_details[sender]["banks"] and user_details[sender]["balance"] >= amount:
                    user_details[sender]["balance"] -= amount
                    final_answer[command_str] = "SUCCESS"
                else:
                    final_answer[command_str] = "FAILURE"
            # Sender is a bank, receiver is a user
            elif sender not in user_details and receiver in user_details:
                if sender in user_details[receiver]["banks"]:
                    user_details[receiver]["balance"] += amount
                    final_answer[command_str] = "SUCCESS"
                else:
                    final_answer[command_str] = "FAILURE"
            else:
                final_answer[command_str] = "FAILURE"

        elif command[0] == "GET":
            name = command[2]
            if name in user_details:
                final_answer[command_str] = f"SUCCESS {user_details[name]['balance']}"
            else:
                final_answer[command_str] = "FAILURE"

    return final_answer

if __name__ == '__main__':
    # Example test input
    input_commands = [
        "INIT,John,1000,BankA,BankB",
        "INIT,Alice,500,BankA",
        "POST,1,John,Alice,200",
        "POST,2,John,BankA,300",
        "GET,3,Alice",
        "GET,4,John",
        "POST,5,Alice,John,1000"  # This should fail due to insufficient balance
    ]
    final_answer = account_manager(input_commands)

    # Print results only for processed POST and GET commands
    for command in input_commands:
        if command.split(",")[0].upper() == "INIT":
            continue
        else:
            print(f"{command} | {final_answer[command]}")
