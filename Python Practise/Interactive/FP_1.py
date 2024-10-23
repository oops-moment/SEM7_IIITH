class AccountManager:
    
    def __init__(self):
        self.init_commands = []
        self.other_commands = []
        self.command2answer = {}
        self.userDetails = {}
        
    def split_command(self, command):
        return command.split(",")

    def process_init(self):
        for command in self.init_commands:
            command = self.split_command(command)
            name = command[1]
            balance = int(command[2])
            banks = set(command[3:])  # Using a set for better performance

            self.userDetails[name] = {
                "balance": balance,
                "banks": banks
            }

    def process_post(self, command: str):
        store_command = command
        command = self.split_command(command)
        sender = command[2]
        receiver = command[3]
        amount = int(command[4])

        if sender in self.userDetails and receiver in self.userDetails:
            if amount <= self.userDetails[sender]["balance"]:
                self.userDetails[sender]["balance"] -= amount
                self.command2answer[store_command] = "SUCCESS"
            else:
                self.command2answer[store_command] = "FAILURE"
        else:
            self.command2answer[store_command] = "FAILURE"

    def process_get(self, command: str):
        name = command.split(",")[2]
        if name in self.userDetails:
            self.command2answer[command] = str(self.userDetails[name]["balance"])
        else:
            self.command2answer[command] = "FAILURE"

    def process_add_bank(self, command: str):
        command = self.split_command(command)
        name, bank = command[2], command[3]

        if name in self.userDetails:
            if bank not in self.userDetails[name]["banks"]:
                self.userDetails[name]["banks"].add(bank)
                self.command2answer[command] = "SUCCESS"
            else:
                self.command2answer[command] = "FAILURE"
        else:
            self.command2answer[command] = "FAILURE"

    def process_remove_bank(self, command: str):
        command = self.split_command(command)
        name, bank = command[2], command[3]

        if name in self.userDetails and bank in self.userDetails[name]["banks"]:
            self.userDetails[name]["banks"].remove(bank)
            self.command2answer[command] = "SUCCESS"
        else:
            self.command2answer[command] = "FAILURE"

    def process_commands(self, input_commands: list[str]):
        for command in input_commands:
            if command.startswith("INIT"):
                self.init_commands.append(command)
            else:
                self.other_commands.append(command)

        self.process_init()
        self.other_commands = sorted(self.other_commands, key=lambda x: int(x.split(',')[1]))

        for command in self.other_commands:
            if command.startswith("POST"):
                self.process_post(command)
            elif command.startswith("GET"):
                self.process_get(command)
            elif command.startswith("ADD_BANK"):
                self.process_add_bank(command)
            else:
                self.process_remove_bank(command)

        return self.command2answer


if __name__ == '__main__':
    N = int(input("Enter the number of commands: "))
    input_commands = [input("Enter the command: ") for _ in range(N)]
    
    accountManager = AccountManager()
    command2answer = accountManager.process_commands(input_commands)
    
    final_answer = [command2answer[cmd] for cmd in input_commands if not cmd.startswith("INIT")]
    print(",".join(final_answer))
