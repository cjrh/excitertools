from excitertools import Iter

x = (
   Iter
      .open('data.txt')
      .map(str.rstrip)  # Remove the trailing newline on each line
      .map(int)         # Convert the data to int
      .filter(lambda v: v % 3 == 0)  # Only allow certain values
      .collect()
)
print(x)
