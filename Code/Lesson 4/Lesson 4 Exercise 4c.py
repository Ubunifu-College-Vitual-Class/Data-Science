# Design a multiple-choice question bank.
# Each question has four options.
# The user needs to pick the correct answer among these options.
# Once the test is over, he can see the score.



# I use a dictionary to store answers to the questions. I also use the len()
# function to get the total number of questions in this question bank. To
# beautify the output, I print the questions and corresponding options in the
# new lines. Here is the complete implementation for you.

# All questions
question1 = "Q1.What is the value of the expression:2*3-4?" \
"\n(a)1" \
"\n(b)2" \
"\n(c)3" \
"\n(d)None."
question2 = "\nQ2.What is the value of the expression:1+(2*3)/4?" \
"\n(a)1.5" \
"\n(b)3" \
"\n(c)3.5" \
"\n(d)None."
question3 = "\nQ3.The set data type can hold duplicate values." \
"The statement is:" \
"\n(a)False" \
"\n(b)True" \
"\n(c)Partially correct." \
"\n(d)None."
# Storing the questions with answer keys
# inside the following dictionary.
question_bank = {question1: "b",
question2: "c",
question3: "a"}
print("Welcome to the Ubunifu College test.")
print("="*25)
score = 0 # initial value
for key in question_bank:
    print(key)
user_input = input("Type your answer(a/b/c/d):")
if user_input == question_bank[key]:
    score += 1
print(f"\nYour Score:{score} out of {len(question_bank)}")

#to do ***check if the correct answer has been given**
