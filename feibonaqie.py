
def feibo():
     a=0
     b=1
     yield b
     while True:
          a, b = b, a+b
          yield b
