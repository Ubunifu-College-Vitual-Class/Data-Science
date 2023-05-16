#Given a Python list, write a program to remove all occurrences of item 20.
#Solution 1
list1 = [5, 20, 15, 20, 25, 50, 20]
def remove_value(sample_list, val):
    return [i for i in sample_list if i != val]
res = remove_value(list1, 20)
print(res)



#Solution 2
while 20 in list1:
    list1.remove(20)
print(list1)
