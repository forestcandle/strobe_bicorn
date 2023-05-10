from utils import *
import math
import random
import sympy

SEED_ONE = 13
SEED_TWO = 18
random.seed(0)

def setup(n, t, alg):
    p_dash = get_germain_prime(SEED_ONE)
    q_dash = get_germain_prime(SEED_TWO) #set seeds just to compare speed for plotting fairly
    #p_dash = get_germain_prime(random.randint(1,int(n)))
    #q_dash = get_germain_prime(random.randint(1,int(n)))
    p = 2*p_dash + 1
    q = 2*q_dash + 1
    N = p*q 

    if alg == "bicorn":

        noMatch = True
        gen = 0

        while noMatch:
            gen_pre = random.randint(0, N)
            first_check = gen_pre % p
            sec_check = gen_pre % q
            if first_check != 0 and first_check != 1 and first_check != p-1 and sec_check != 0 and sec_check != 1 and sec_check != q-1:
                gen = pow(gen_pre, 2, N)
                noMatch = False

        assert(noMatch == False)

        out_h = gen**2**t
        #rangee = pow(p_dash * q_dash,1,t)+n
        rangee = SEED_ONE * SEED_TWO
        #print("N, gen, out_h, rangee: ", N, gen, out_h, rangee)

        return N, gen, out_h, rangee

    elif alg == "strobe":

        phi = 4*p_dash*q_dash
        # randomly sample a prime s from the range [n+1, min(p_dash, q_dash)-1]
        s = sympy.randprime(n+1, min(p_dash, q_dash)-1)

        try:
            assert(s > n)
            assert(s < min(p_dash, q_dash))
            assert(phi % s != 0)
        except AssertionError:
            print("s is not valid")
            return None
        
        v = sympy.mod_inverse(math.factorial(n)*s, phi//4)
        a_coeff = [sympy.randprime(1, N) for _ in range(t-1)]

        return N, phi, s, a_coeff

    else:
      print("Algorithm not implemented")
      return 
    

def gen(n_fact, N):
    seed = random.randint(1, N)
    n_fact_sq = n_fact**2
    x_0 = pow(seed, n_fact_sq, N)
    return x_0
