import random

class RandomizedSet:
    def __init__(self):
        self.s = set()

    def insert(self, val: int) -> bool:
        if val in self.s:  # Fix: Checking if val is already in the set
            return False
        
        self.s.add(val)
        return True
    
    def remove(self, val: int) -> bool:
        if val not in self.s:  # Fix: Correctly checking if val is absent
            return False
        
        self.s.remove(val)
        return True
    
    def getRandom(self) -> int:
        if self.s:  # Ensure there are elements in the set before getting a random element
            return random.choice(list(self.s))
        return -1 
    
if __name__ == '__main__':
    final_answer = []
    solutionClass = RandomizedSet()

    input_commands = []
    input_command_values = []

    N = int(input("Enter the number of commands: "))
    
    # Input the command names
    for _ in range(N):
        command = input("Enter command: ")
        input_commands.append(command)

    # Input the command values
    for _ in range(N):
        value = int(input("Enter value (if any, else leave blank): ") or -1)  
        input_command_values.append([value] if value != -1 else [])
    
    for i in range(N):
        if input_commands[i] == "RandomizedSet":
            solutionClass.__init__()  
        elif input_commands[i] == "insert":
            final_answer.append(solutionClass.insert(input_command_values[i][0]))
        elif input_commands[i] == "remove":
            final_answer.append(solutionClass.remove(input_command_values[i][0]))
        elif input_commands[i] == "getRandom":
            final_answer.append(solutionClass.getRandom())
    
    print(final_answer)  