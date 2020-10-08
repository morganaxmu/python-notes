#Task 1 Duplicate Removal
def dure(L):
    proxy = set()
    for i in L:
        proxy.add(i)
    return list(proxy)

L1 = [3,7,11,-2,7,-7,1,13,7,2,3,1]
L2 = ["hi","Rob","Steven","hello","hello world","hi world","world","Rob","hello","hello Steven"]
L3 = [2,"hi",-2,"2",0,"=","Rob","0",2.0,0]

dure(L1)
