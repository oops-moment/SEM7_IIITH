# Introduction to Python Basics
# Examples of Python usage in popular platforms: YouTube, Instagram, and Spotify are all programmed in Python.

# Printing "Hello World" to the console
print("Hello World")

# Declaring variables
age = 20
price = 19.95

# Printing variables
print(age)   # Output: 20
print(price)  # Output: 19.95

# String variables
first_name = "Prisha"
last_name = "Arora"

# Boolean variable
bool_value = True

# Reassigning variables
first_name = "John"
last_name = "Smith"
age = 20
new_patient = True

# Uncomment below lines for input and concatenation example:
# name = input("What is the patient's name? ")
# print("Patient's name: " + name)

# Data types: numbers, strings, booleans
# Taking user input (always stored as string)
birth_year = input("Enter your birth year: ")
print(type(birth_year))  # Output: <class 'str'>

# Converting birth year to integer and calculating age
age = 2024 - int(birth_year)
print(age)  # Output: Calculated age

# Taking two numeric inputs and adding them
first_number = input("Enter the first number: ")
second_number = input("Enter the second number: ")

# Converting input to float for mathematical operations
first_number = float(first_number)
second_number = float(second_number)

# Printing the sum of the two numbers
print("Additive value: " + str(first_number + second_number))

# Working with strings
course = "Python for beginners"

# String methods
print(course.upper())  # Converts string to uppercase
print(course.lower())  # Converts string to lowercase
print(course)          # Original string remains unchanged

# Finding a character in a string
print(course.find('t'))  # Returns index of first occurrence of 't'

# Replacing part of a string
print(course.replace('for', '4'))  # Replaces 'for' with '4'
print(course.replace('g', '0'))    # Replaces 'g' with '0'

# Checking for a substring within a string
print('Python' in course)  # Checks if 'Python' is in the string

# Arithmetic operations
print(10 / 3)  # Division: returns 3.333...
print(10 // 3) # Integer division: returns 3

# Augmented assignment
X = 10
X += 3  # Same as X = X + 3
X -= 3  # Same as X = X - 3
print(X)  # Output: 10

# Operator precedence
X = 10 + 3 * 2  # Multiplication happens first
print(X)  # Output: 16

X = (10 + 3) * 2  # Parentheses change the precedence
print(X)  # Output: 26

# Comparison operators
print(3 > 2)   # True
print(3 == 2)  # False
print(3 != 2)  # True

# Logical operators
X = 30
print(not (X > 10 and X < 40))  # True, because both conditions are True

# if-else statement
if True:
    print("yo")
    print("yo2")

# While loop example
i = 1
while i <= 5:
    print(i * '*')  # Prints '*' multiplied by the current value of i
    i += 1

# Working with lists
names = ['Prisha', 'Laksh', 'Chicl00']

# List slicing
print(names[:1])  # ['Prisha']
print(names[1:])  # ['Laksh', 'Chicl00']
print(names[0:2]) # ['Prisha', 'Laksh']

# More list operations
numbers = [1, 2, 3, 2]
print(numbers)

# Insert value 10 at index 2
numbers.insert(2, 10)
print(numbers)  # [1, 2, 10, 3, 2]

# Checking if 10 exists in the list
print(10 in numbers)  # True

# Iterating over a list
for number in numbers:
    print(number)

# Working with range
numbers = range(5)
print(numbers)  # range(0, 5)

# Range with start, stop, and step
numbers = range(5, 10, 3)
for num in numbers:
    print(num)  # Outputs 5 and 8