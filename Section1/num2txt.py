# Helper Function to convert from 0 (zero) to 99999 (ninety-nine thousand, nine hundred and ninety-nine
def numbertowords(num):
    # Edge Cases
    if num > 99999:
        return str(num)  # If the number exceeds 99999, just return the number itself as a string
    
    if num == 0:
        return "zero"
    
    first_twenty = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    def convert_chunk(n):
        # Convert a number less than 1000 to words
        if n == 0:
            return ""
        elif n < 20:
            return first_twenty[n]
        elif n < 100:
            return tens[n // 10] + ("" if n % 10 == 0 else "-" + first_twenty[n % 10])  # 99 -> ninety-nine
        else:
            return first_twenty[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + convert_chunk(n % 100))  # 999 -> nine hundred and ninety-nine
    
    if num < 1000:
        return convert_chunk(num)
    else:
        # Recursive, if num >= 1000
        thousands = num // 1000
        remainder = num % 1000
        if remainder == 0:
            return convert_chunk(thousands) + " thousand"
        elif remainder < 100: 
            return convert_chunk(thousands) + " thousand and " + convert_chunk(remainder)
        else:
            return convert_chunk(thousands) + " thousand, " + convert_chunk(remainder)


def Conversion(sentence):
    words = sentence.split()  # Splits the string into a list of words
    result = []

    for word in words:
        if word.isdigit():
            num = int(word)
            if num <= 99999:
                result.append(numbertowords(num)) 
            else:
                result.append(word)  # Exceeds 99999, out of range
        else: 
            result.append(word)

    return ' '.join(result)

#Test Cases
print(Conversion("The painting costs 10001 dollars."))
print(Conversion("There are 92341 reasons why you will succeed."))
print(Conversion("153"))
print(Conversion("99999"))
#O(N) Time Complexity