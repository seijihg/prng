import numpy as np
from seed import example_seed


class PseudoRandomGenerator(object):
    """Base class for any Pseudo-Random Number Generator."""

    def __init__(self, X0=0):
        """Create a new PRNG with seed X0."""
        self.X0 = X0
        self.X = X0
        self.t = 0
        self.max = 0

    def __iter__(self):
        """self is already an iterator!"""
        return self

    def seed(self, X0=None):
        """Reinitialize the current value with X0, or self.X0.

        - Tip: Manually set the seed if you need reproducibility in your results.
        """
        self.t = 0
        self.X = self.X0 if X0 is None else X0

    def __next__(self):
        """Produce a next value and return it."""
        # This default PRNG does not produce random numbers!
        self.t += 1
        return self.X

    def randint(self, *args, **kwargs):
        """Return an integer number in [| 0, self.max - 1 |] from the PRNG."""
        return self.__next__()

    def int_samples(self, shape=(1,)):
        """Get a numpy array, filled with integer samples from the PRNG, of shape = shape."""
        # return [ self.randint() for _ in range(size) ]
        return np.fromfunction(np.vectorize(self.randint), shape=shape, dtype=int)

    def rand(self, *args, **kwargs):
        """Return a float number in [0, 1) from the PRNG."""
        return self.randint() / float(1 + self.max)

    def float_samples(self, shape=(1,)):
        """Get a numpy array, filled with float samples from the PRNG, of shape = shape."""
        # return [ self.rand() for _ in range(size) ]
        return np.fromfunction(np.vectorize(self.rand), shape=shape, dtype=int)


class MersenneTwister(PseudoRandomGenerator):
    """The Mersenne twister Pseudo-Random Number Generator (MRG)."""

    def __init__(
        self,
        seed=None,
        w=32,
        n=624,
        m=397,
        r=31,
        a=0x9908B0DF,
        b=0x9D2C5680,
        c=0xEFC60000,
        u=11,
        s=7,
        v=15,
        l=18,
    ):
        """Create a new Mersenne twister PRNG with this seed."""
        self.t = 0
        # Parameters
        self.w = w
        self.n = n
        self.m = m
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.u = u
        self.s = s
        self.v = v
        self.l = l
        # For X
        self.X0 = seed
        self.X = np.copy(seed)
        # Maximum integer number produced is 2**w - 1
        self.max = (1 << w) - 1

    def __next__(self):
        """Produce a next value and return it, following the Mersenne twister algorithm."""
        self.t += 1
        # 1. --- Compute x_{t+n}
        # 1.1.a. First r bits of x_t : left = (x_t >> (w - r)) << (w - r)
        # 1.1.b. Last w - r bits of x_{t+1} : right = x & ((1 << (w - r)) - 1)
        # 1.1.c. Concatenate them together in a binary vector x : x = left + right
        left = self.X[0] >> (self.w - self.r)
        right = self.X[1] & ((1 << (self.w - self.r)) - 1)
        x = (left << (self.w - self.r)) + right
        xw = x % 2  # 1.2. get xw
        if xw == 0:
            xtilde = x >> 1  # if xw = 0, xtilde = (x >> 1)
        else:
            xtilde = (x >> 1) ^ self.a  # if xw = 1, xtilde = (x >> 1) ⊕ a
        nextx = self.X[self.m] ^ xtilde  # 1.3. x_{t+n} = x_{t+m} ⊕ \tilde{x}
        # 2. --- Shift the content of the n rows
        self.X[:-1] = self.X[
            1:
        ]  # 2.a. shift one index on the left, x1..xn-1 to x0..xn-2
        self.X[-1] = nextx  # 2.b. write new xn-1
        # 3. --- Then use it to compute the answer, y
        y = nextx  # 3.a. y = x_{t+n}

        y ^= y >> self.u  # 3.b. y = y ⊕ (y >> u)
        y ^= (y << self.s) & self.b  # 3.c. y = y ⊕ ((y << s) & b)
        y ^= (y << self.v) & self.c  # 3.d. y = y ⊕ ((y << v) & c)
        y ^= y >> self.l  # 3.e. y = y ⊕ (y >> l)
        return y


result = MersenneTwister(seed=example_seed)
print(result.int_samples((1000,)))
