import math
choose = math.comb

def mac_to_int(cs):
    return sum(choose(c,i+1) for i,c in enumerate(sorted(cs)))

def avg(a,b):
    return (a+b) // 2

# https://chat.openai.com/share/75e92538-7be6-404e-b3f9-83da7d845c27
def calculate_bounds_old(k, X):
    term1 = (2 * math.pi * k) ** (1 / (2 * k))
    term2 = k / math.e
    term3 = X ** (1 / k)

    upper_e_term = math.exp(1 / (12 * k) / k)
    lower_e_term = math.exp(1 / (12 * k + 1) / k)

    shared_factor = term1 * term2 * term3

    lower_bound = math.floor(shared_factor * lower_e_term)
    upper_bound = math.ceil(shared_factor * upper_e_term + k + 1)

    return lower_bound, upper_bound

import math

def calculate_bounds_new(k, X):

    t1 = math.pi ** (1 / (2 * k))
    t2 = k / math.e
    t3 = X ** (1 / k)

    pre_t4_common = 8 * k ** 3 + 4 * k**2 + k
    pre_t4_lower = pre_t4_common + 1/100
    pre_t4_upper = pre_t4_common + 1/30

    t4_lower = pre_t4_lower ** (1 / (6 * k))
    t4_upper = pre_t4_upper ** (1 / (6 * k))

    lower_bound = math.floor(t1 * t2 * t3 * t4_lower)
    upper_bound = math.ceil(t1 * t2 * t3 * t4_upper + k)

    # Return the bounds
    return lower_bound, upper_bound

def ramanujan_log_factorial(n):
    return n*math.log(n) - n + 1/6 * math.log( 8 * n**3 + 4 * n**2 + n + 1/30 ) + 1/2 * math.log(math.pi)

def approx_log_combo(n, k):
    if n < k:
        return -math.inf
    if k == 0 or k == n:
        return 0
    return ramanujan_log_factorial(n) - ramanujan_log_factorial(k) - ramanujan_log_factorial(n-k)

calculate_bounds = calculate_bounds_new

from tqdm import tqdm

small_threshold = 1_000

# use https://math.stackexchange.com/questions/3164173/inverse-reverse-of-number-of-permutations-and-of-number-of-combinations-with-rep
# I corrected the hallucinated ChatGPT suggestions a bit--I'm using these upper and lower Stirling + combinations bounds for numeric stability for high numbers.
# results in O(k log k) asymptotic time complexity - the same as mac_to_int!
def int_to_mac(int_, k,use_tqdm=False):
    csDecreasing = []

    currInt = int_

    rng = range(k, 0, -1)
    if use_tqdm:
        rng = tqdm(rng)
    for i in rng:
        nextInclMinNew, nextExclMaxNew = calculate_bounds_new(i, currInt)
        nextInclMinOld, nextExclMaxOld = calculate_bounds_old(i, currInt)

        # print log of the difference between the two bounds
        # print(f"Log of difference between upper and lower bound for new method: {math.log(nextExclMaxNew-nextInclMinNew)}")
        # print(f"Log of difference between upper and lower bound for old method: {math.log(nextExclMaxOld-nextInclMinOld)}")

        nextInclMin, nextExclMax = calculate_bounds_old(i, currInt)

        # assert choose(nextInclMin,i) <= currInt and choose(nextExclMax,i) > currInt,f"nextInclMin={nextInclMin}, nextExclMax={nextExclMax}, currInt={currInt}, i={i}"

        # as individual asserts:
        # assert choose(nextInclMin,i) <= currInt,f"nextInclMin={nextInclMin}, currInt={currInt}, i={i}"
        # assert choose(nextExclMax,i) > currInt,f"nextExclMax={nextExclMax}, currInt={currInt}, i={i}"

        if currInt == 0:
            if len(csDecreasing) == 0:
                return range(k)
            else:
                csDecreasing.append(min(csDecreasing[-1],i)-1)
            continue

        curr_log_int = math.log(currInt)

        while nextInclMin + 1 < nextExclMax:
            currAvg = avg(nextExclMax, nextInclMin)

            if currAvg < small_threshold:
                currCombo = choose(currAvg, i)

                if currCombo > currInt:
                    nextExclMax = currAvg
                else:
                    nextInclMin = currAvg
            else:
                currLogCombo = approx_log_combo(currAvg, i)

                if currLogCombo > curr_log_int:
                    nextExclMax = currAvg
                else:
                    nextInclMin = currAvg

        c = nextInclMin

        # assert choose(c,i) <= currInt and choose(c+1,i) > currInt,f"C is incorrect: ({c} choose {i}) = {choose(c,i)} <= {currInt}"

        # if c < small_threshold and (len(csDecreasing) == 0 or csDecreasing[-1] >= small_threshold):
        #     print("crossed small-c threshold")
        csDecreasing.append(c)
        currInt -= choose(c,i)


    csDecreasing.reverse()
    return csDecreasing

def checkIdentity(int_, k):
    assert mac_to_int(int_to_mac(int_, k))==int_, f"Invalid identity with int={int_},k={k}"

if __name__ == "__main__":
    from cProfile import Profile
    from pstats import SortKey, Stats

    print("Testing...")
    # with Profile() as profile:

    from time import time
    start_time = time()
    checkIdentity(9_000_000_000_000_000,400_000)
    print(f"Time taken: {time()-start_time} seconds")
    #     print(
    #          Stats(profile)
    #             .strip_dirs()
    #             .sort_stats(SortKey.CALLS)
    #             .print_stats()
    #     )
    # print("Big identity test passed!")

    print("Testing small identities...")
    for k in tqdm(range(2,300)):
        for i in range(300):
            checkIdentity(i, k) # succeeds!
    print("Small identity test passed!")