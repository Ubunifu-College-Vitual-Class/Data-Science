#1) Create a list that contains more than 2 elements. Then remove the last two elements from the list
my_list = ["John", 12, 25, 12,"Sam", True, 50.7]
print("The original list is:")
print(my_list)
#Removing the last two elements from the list
del(my_list[-2:])
print("Now the list is:")
print(my_list)


#2) Create a tuple with more than 5 elements. Then print the 3rd element of it. Check if you can you print the 3rd element from last?
my_tuple = ("John", 12, 25, 12, "Sam", True, 50.7)
print("The original tuple is:")
print(my_tuple)
#Printing the 3 element
print("The third element:")
print(my_tuple[2])
print("The third element from last:")
print(my_tuple[-3])


#3) Create a tuple with elements 1,2,2,3,4,4,4,5.Then reverse the tuple and print how many times 4 appears in this tuple.
my_tuple = (1, 2, 2, 3, 4, 4, 4, 5)
print("The original tuple is:")
print(my_tuple)
print("The reversed tuple is:")
rev_tuple = tuple(reversed(my_tuple))
print(rev_tuple)
print(f"The number of 4 in this tuple:{rev_tuple.count(4)}")

#4) Create a dictionary and print the details. Also print the value for a particular key.
my_dictionary = {1: "John", 2: 12, 3: "Sam", 4: True, 5: 50.7}
print("The my_dictionary contains:")
print(my_dictionary)
print("Value at key 1:", my_dictionary[1])
print("Value at key 2:", my_dictionary[2])


