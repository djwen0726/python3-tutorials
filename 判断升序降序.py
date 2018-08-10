
def which_order(lst):
     print(list(lst))
     if sorted(lst) == list(lst):
          return '升序'
     elif sorted(lst, reverse=True) == list(lst):
          return '降序'
     else:
          return '无序'


