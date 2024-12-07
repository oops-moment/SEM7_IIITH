# Initialize an array to store log messages
log_messages = []

# Function to calculate the sum of squares and log messages
def sum_of_squares(n):
    total = 0
    for i in range(1, n + 1):
        square = i ** 2
        total += square
        log_messages.append(f"Calculating square of {i}: {square}, running total: {total}")
    return total

# Call the function
result = sum_of_squares(5)

# Add final result to log
log_messages.append(f"Final sum of squares: {result}")

# Write log messages to a file
with open('log_output.txt', 'w') as log_file:
    for message in log_messages:
        log_file.write(message + '\n')

print("Calculation complete. Log written to 'log_output.txt'.")