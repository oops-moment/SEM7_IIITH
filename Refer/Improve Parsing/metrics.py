import re
from collections import defaultdict

def parse_metrics(file_path):
    # Dictionary to store the total time taken for each function
    function_times = defaultdict(float)

    # Regex to extract function name and time taken
    pattern = re.compile(r"Time taken to execute (\w+): (\d+\.\d+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                function_name = match.group(1)
                time_taken = float(match.group(2))
                function_times[function_name] += time_taken

    return function_times

def display_total_times(function_times):
    result = "Overall Conclusion:\n"
    for function_name, total_time in function_times.items():
        result += f"{function_name}: {total_time:.6f} seconds\n"
    return result

def append_to_file(file_path, conclusion_text):
    with open(file_path, 'a') as file:
        file.write("\n" + conclusion_text)

# Replace with the path to your metrics.txt file
file_path = "metrics_th_old.txt"

function_times = parse_metrics(file_path)
conclusion_text = display_total_times(function_times)
append_to_file(file_path, conclusion_text)

print("Results appended to metrics.txt successfully.")