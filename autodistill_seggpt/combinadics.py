import math
choose = math.comb

def mac_to_int(cs):
    return sum(choose(c,i+1) for i,c in enumerate(sorted(cs)))

def avg(a,b):
    return (a+b) // 2

def int_to_mac(int_, k):

    topInclMin = 0

    while choose(topInclMin,k) <= int_:
        topInclMin = topInclMin * 2 + 1  # to bring 0 -> 1

    nextExclMax = topInclMin

    csDecreasing = []

    currInt = int_

    for i in range(k, 0, -1):
        nextInclMin = 0
        
        while nextInclMin + 1 < nextExclMax:
            currAvg = avg(nextExclMax, nextInclMin)
            currCombo = choose(currAvg, i)

            if currCombo > currInt:
                nextExclMax = currAvg
            else:
                nextInclMin = currAvg

        c = nextInclMin

        assert choose(c,i) <= currInt and choose(c+1,i) > currInt,f"C is incorrect: ({c} choose {i}) = {choose(c,i)} <= {currInt}"

        csDecreasing.append(c)
        currInt -= choose(c,i)

    csDecreasing.reverse()
    return csDecreasing

def checkIdentity(int_, k):
    assert mac_to_int(int_to_mac(int_, k))==int_, f"Invalid identity with int={int_},k={k}"


if __name__ == "__main__":
    for k in range(2,30):
        for i in range(300):
            checkIdentity(i, k)