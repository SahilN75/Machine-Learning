#assignment 1 (ML)

import random
import statistics

def pairs_with_10(numbers):  #pairs with 10 counts how many pairs in those list satisfy them
    count = 0
    length = len(numbers)
    for i in range(length):   #i is the first element of the pair
        for j in range(i+1,length):  #i+1 is to avoid repeating the pairs
            if numbers[i] + numbers[j] == 10:
                count += 1

        return count

def find_range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    return max(numbers) - min(numbers)

def matrix_power(matrix, power):
    size = len(matrix) #stores the number of rows square matrix so its rows=coloumns

    result = [] #Empty list to store the identity matrix
    for i in range(size):
        row = [] #my current row
        for j in range(size):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        result.append(row)

    # Function to multiply two matrices
    def multiply_matrices(matrix1, matrix2):
        product = []
        for i in range(size):
            row = []
            for j in range(size):
                value = 0
                for k in range(size):
                    value = value + matrix1[i][k] * matrix2[k][j]
                row.append(value)
            product.append(row)
        return product

    # Multiply matrix 'power' number of times
    for _ in range(power):
        result = multiply_matrices(result, matrix)

    return result

def highest_occurring_character(text):
    frequency = {}
    for char in text:
        if char.isalpha(): #to check if its an alphabet
            char = char.lower() #converting text to lowercase to avoid mismatch
            frequency[char] = frequency.get(char, 0) + 1

    max_char = max(frequency, key=frequency.get) #.get to find the highest frequency char
    return max_char, frequency[max_char]



def calculate_statistics(numbers):
    mean_value = statistics.mean(numbers)
    median_value = statistics.median(numbers)
    try:
        mode_value = statistics.mode(numbers) #most frequent value
    except statistics.StatisticsError:
        mode_value = "No unique mode" # if not single most frequent value
    return mean_value, median_value, mode_value




#Main program 
#for 1st question to input my list and find the pairs with sum 10
numbers_list = [2, 7, 4, 1, 3, 6]
my_pair = pairs_with_10(numbers_list)
print("Pairs with sum 10:", my_pair)

#i user inputted to check for list length lesser than 3 also if its higher to check my range
numbers = list(map(int, input("Enter the list").split())) 
print(find_range(numbers))

#Question 3
n = int(input("Enter matrix size: "))
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)
    
m = int(input("Enter power: "))
output = matrix_power(A, m)
print("Result matrix:")
for row in output:
    print(row)

#Inputing my string 
input_string = input("Enter a string: ")
character, count = highest_occurring_character(input_string)
print("Highest occurring character:", character)
print("Occurrence count:", count)

#imported random to generate numbers
random_numbers = [random.randint(1, 10) for _ in range(25)] #to generate numbers from 1 to 10 maximum 25 numbers
mean_val, median_val, mode_val = calculate_statistics(random_numbers)
print("Random numbers:", random_numbers) #random 25 numbers generated
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)

