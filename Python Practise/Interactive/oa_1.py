class Solution:
    def solve(self, commands: list[str]) -> list[str]:
        init_commands = []
        other_commands = []
        
        # Separate INIT commands from other commands
        for command in commands:
            if command.split(',')[0] == "INIT":
                init_commands.append(command)
            else:
                other_commands.append(command)
        
        # Sort other commands by their timestamp (2nd element)
        other_commands = sorted(other_commands, key=lambda x: int(x.split(',')[1]))
        
        # Initialize user details based on INIT commands
        user_details = {}
        for command in init_commands:
            parts = command.split(',')
            name = parts[1]
            balance = int(parts[2])
            banks = parts[3:]
            user_details[name] = {
                'balance': balance,
                'banks': banks
            }
        
        # Process other commands (GET and POST)
        answers = {}
        for command in other_commands:
            original_command = command  # Keep a reference to the original command string
            parts = command.split(',')
            
            if parts[0] == "GET":
                # GET command
                name = parts[2]
                if name not in user_details:
                    answers[original_command] = "FAILURE"
                else:
                    answers[original_command] = str(user_details[name]['balance'])
            else:
                # POST command
                sender = parts[2]
                receiver = parts[3]
                amount = int(parts[4])

                if sender in user_details and receiver in user_details:
                    if user_details[sender]['balance'] < amount:
                        answers[original_command] = "FAILURE"
                    else:
                        answers[original_command] = "SUCCESS"
                        user_details[sender]['balance'] -= amount
                        user_details[receiver]['balance'] += amount
                elif sender not in user_details:
                    if sender not in user_details[receiver]['banks']:
                        answers[original_command] = "FAILURE"
                    else:
                        answers[original_command] = "SUCCESS"
                        user_details[receiver]['balance'] += amount
                elif receiver not in user_details:
                    if receiver not in user_details[sender]['banks'] or user_details[sender]['balance'] < amount:
                        answers[original_command] = "FAILURE"
                    else:
                        answers[original_command] = "SUCCESS"
                        user_details[sender]['balance'] -= amount
        
        return answers

if __name__ == '__main__':
    number_commands = int(input("Enter number of commands: "))
    commands_input = []
    
    for _ in range(number_commands):
        command = input("Enter the command: ")
        commands_input.append(command)
    
    solution = Solution()
    answers = solution.solve(commands_input)
    
    final_answers = []
    for command in commands_input:
        if command.split(',')[0] == "INIT":
            continue
        final_answers.append(answers[command])
    
    print(",".join(final_answers))
