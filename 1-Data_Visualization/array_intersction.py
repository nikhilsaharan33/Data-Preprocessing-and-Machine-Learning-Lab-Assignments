def intersection(arr1, arr2, n, m):
    def merge(arr1, arr2):
        a = []
        l1 = len(arr1)
        l2 = len(arr2)
        start1 = 0
        start2 = 0
        while start1 < l1 and start2 < l2:
            if arr1[start1] <= arr2[start2]:
                a.append(arr1[start1])
                start1 += 1
            elif arr1[start1] > arr2[start2]:
                a.append(arr2[start2])
                start2 += 1
        while start1 < l1:
            a.append(arr1[start1])
            start1 += 1
        while start2 < l2:
            a.append(arr2[start2])
            start2 += 1
        return a

    def mergesort(a):
        n = len(a)
        if n == 0 or n == 1:
            return a
        return merge(mergesort(a[:n // 2]), mergesort(a[n // 2:]))
    def binarysearch(arr, x):
        n=len(arr)
        s=0
        e=n-1
        while s<e:
            mid=(s+e)//2
            ele=arr[mid]
            if ele == x:
                arr[mid]=-100000000000000000
                return ele
            elif ele>=x:
                e=mid-1
            else:
                s=mid+1
        else:
            return -100000000000000000


    a1=mergesort(arr1)
    a2=mergesort(arr2)
    s=''
    for i in range(n):
        if binarysearch(a2,a1[i]) != -100000000000000000:
            s=s+str(binarysearch(a2,a1[i]))+' '
    l=len(s)
    if l != 0:
        s1=s[:(l-1)]
        return s1
    else:
        return s







