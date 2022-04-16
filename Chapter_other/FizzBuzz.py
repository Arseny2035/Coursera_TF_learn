
# for i in range(1, 101):
#     if (i % 3 == 0):
#         if i % 5 == 0:
#             print("FizzBuzz")
#         else:
#             print("Fizz")
#     else:
#         if i % 5 == 0:
#             print("Buzz")
#         else:
#             print(i)


# for i in range(1, 101):
#     str = ""
#     if i % 3 == 0:
#         str += "Fizz"
#     if i % 5 == 0:
#         str += "Buzz"
#     print(str if str else i)

print('\n'.join(['Fizz'*(x % 3 == 2) + 'Buzz'*(x % 5 == 4) or str(x + 1) for x in range(100)]))