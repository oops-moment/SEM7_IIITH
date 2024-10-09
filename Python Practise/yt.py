# Youtube , instagram, spotify all programmed in python

print("Hello World")

#  declaring variables
age = 20
price = 19.95
print(age)
print(price)

first_name = "Prisha"
last_name = "Arora"

bool_value= True

first_name ="John"
last_name = "Smith"

age= 20

new_patient = True

# name = input("What is the patients name ? ")
# print("name" + name)

#  three types of data : number , string , boolean
birth_year = input("Enter your birth year: ")  #-->always a string 
print(type(birth_year))

age= 2024-int(birth_year)
print(age)

first_number=input("Enter the first number")
second_number= input("Enter the second number")

first_number=float(first_number)
second_number=float(second_number)

print("additive value "+ str(first_number+second_number))

#  strings 

course = "Python for beginners"
#  methods to specific to string

print(course.upper())
print(course)

print(course.find('t'))

print(course.replace('for','4'))

#  note that in all this original string remained unchanged that is that remains immutable irrespective

print(course.replace('g','0'))

print('Python' in course)

print(10/3)

print(10//3)


X = 10
X = X + 3
X += 3
X -= 3

print(X)


#  Operator Precedence

X = 10 + 3*2
print(X)

X=(10+3)*2
print(X)

X= 3 >2
print(X)

X= 3==2
print(X)

#  operators are > >= < <= == !=

# logical operators 

X=30
print(not (X >10 and X<40))

#  and when both expression are true, or whren atleast one expression is truye

if (True):
    print("yo")
    print("yo2")


# concept of while lop

i = 1
while i<=5 : 
    print(i * '*')
    i+=1

names = ['Prisha','Laksh','Chicl00']
# print(names)
# print(names[1])
# print(names[-2])
print(names[:1])
print(names[1:])
print(names[0:2]) #just that you dont include end index , but you include the first index , basically two index 


numbers=[1,2,3,2]
print(numbers)
numbers.insert(2,10)


print(10 in numbers)
for i in numbers:
    print(i)

numbers= range(5)
print(numbers)

numbers = range(5,10,3)

for num in numbers:
    print(num)