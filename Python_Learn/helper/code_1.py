def square(num):
    return num**2

# Simple dictionary
my_dict = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Simple list
my_list = [10, 20, 30, 40, 50]

def simple_iteration_demo():
    print("Iterating over dictionary:")
    for key, value in my_dict.items():
        print(f"Key: {key}, Value: {value}")
    
    print("\nAccessing value from dictionary:")
    print(f"Name: {my_dict['name']}")
    
    # Iterating over a list
    print("\nIterating over list:")
    for item in my_list:
        print(f"List item: {item}")


if __name__=='__main__':
    num=int(input("Enter a number: "))
    print("Square of",num,"is",square(num))
    # Call the function
    simple_iteration_demo()