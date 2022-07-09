def largest_modulo(l,r):
    if l>r//2:
        return r%l
    else:
        a=(r//2+1)
        return r%a
t=int(input())
a=[]
while t>0:

    arr=[int(i) for i in input().split()]
    l,r=arr[0],arr[1]
    a.append(largest_modulo(l,r))
    t-=1
for i in a:
    print(i)

