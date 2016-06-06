
# coding: utf-8

# # Developing Quaternion Tools for iPython3

# In[1]:

from sympy import *
init_session(quiet=True)
import unittest


# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.

# In[2]:

def sr_gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""
    
    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** 0.5

def sr_gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""
    
    g = sr_gamma(beta_x, beta_y, beta_z)
    
    return [g, g * beta_x, g * beta_y, g * beta_z]


# Define a class Qh to manipulate quaternions as Hamilton would have done it so many years ago. Note: I do think we can learn more of the silent workings of Nature using a quaternion on a quaternion manifold, but for now this is much easier to implement.

# In[3]:

class Qh:
    """Quaternions as Hamilton would have defined them, on the manifold R^4."""

    def __init__(self, values=None):
        if values is None:
            self.t, self.x, self.y, self.z = 0, 0, 0, 0
        else:
            self.t, self.x, self.y, self.z = values[0], values[1], values[2], values[3]

    def __str__(self):
        """Customize the output."""
        return "{t}t  {x}i  {y}j  {z}k".format(t=self.t, x=self.x, y=self.y, z=self.z)
        
    def q0(self):
        """Return a zero quaternion."""
        
        return Qh([0, 0, 0, 0])
      
    def q1(self):
        """Return a multiplicative identity quaternion."""
        
        return Qh([1, 0, 0, 0])
    
    def conj(self, type=0):
        """Three types of conjugates."""
        
        t, x, y, z = self.t, self.x, self.y, self.z
        conjq = Qh()

        if type == 0:
            conjq.t = t
            conjq.x = -1 * x
            conjq.y = -1 * y
            conjq.z = -1 * z
        
        if type == 1:
            conjq.t = -1 * t
            conjq.x = x
            conjq.y = -1 * y
            conjq.z = -1 * z
        
        if type == 2:
            conjq.t = -1 * t
            conjq.x = -1 * x
            conjq.y = y
            conjq.z = -1 * z

        return conjq
    
    def commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        s_t, s_x, s_y, s_z = self.t, self.x, self.y, self.z
        q1_t, q1_x, q1_y, q1_z = q1.t, q1.x, q1.y, q1.z

        products = {'tt': s_t * q1_t,
                    'xx+yy+zz': s_x * q1_x + s_y * q1_y + s_z * q1_z,
        
                    'tx+xt': s_t * q1_x + s_x * q1_t,
                    'ty+yt': s_t * q1_y + s_y * q1_t,
                    'tz+zt': s_t * q1_z + s_z * q1_t}
        
        return products
    
    def anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""
        
        s_x, s_y, s_z = self.x, self.y, self.z
        q1_x, q1_y, q1_z = q1.x, q1.y, q1.z

        products = {'yz-zy': s_y * q1_z - s_z * q1_y,
                    'zx-xz': s_z * q1_x - s_x * q1_z,
                    'xy-yx': s_x * q1_y - s_y * q1_x
                    }
        
        return products
    
    def all_products(self, q1):
        """Returns a dictionary with all possible products."""
        
        products = self.commuting_products(q1)
        products.update(self.anti_commuting_products(q1))
        
        return products
    
    def square(self):
        """Square a quaternion."""
        
        qxq = self.commuting_products(self)
        
        sq_q = Qh()
        sq_q.t = qxq['tt'] - qxq['xx+yy+zz']
        sq_q.x = qxq['tx+xt']
        sq_q.y = qxq['ty+yt']
        sq_q.z = qxq['tz+zt']

        return sq_q
    
    def norm(self):
        """The norm of a quaternion."""
        
        qxq = self.commuting_products(self)
        
        n_q = Qh()
        n_q.t = qxq['tt'] + qxq['xx+yy+zz']

        return n_q
    
    def norm_of_vector(self):
        """The norm of the vector of a quaternion."""

        qxq = self.commuting_products(self)
        
        nv_q = Qh()
        nv_q.t = qxq['xx+yy+zz']

        return nv_q
    
    def abs_of_q(self):
        """The absolute value, the square root of the norm."""

        a = self.norm()
        sqrt_t = a.t ** 0.5
        a.t = sqrt_t
        
        return a

    def abs_of_vector(self):
        """The absolute value of the vector, the square root of the norm of the vector."""

        av = self.norm_of_vector()
        sqrt_t = av.t ** 0.5
        av.t = sqrt_t
        
        return av
        
    def add(self, q1):
        """Form a add given 2 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        
        add_q = Qh()
        add_q.t = t1 + t2
        add_q.x = x1 + x2
        add_q.y = y1 + y2
        add_q.z = z1 + z2
                    
        return add_q    

    def dif(self, q1):
        """Form a add given 2 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        
        add_q = Qh()
        add_q.t = t1 - t2
        add_q.x = x1 - x2
        add_q.y = y1 - y2
        add_q.z = z1 - z2
                    
        return add_q
    
    def product(self, q1):
        """Form a product given 2 quaternions."""

        qxq = self.all_products(q1)
        pq = Qh()
        pq.t = qxq['tt'] - qxq['xx+yy+zz']
        pq.x = qxq['tx+xt'] + qxq['yz-zy']
        pq.y = qxq['ty+yt'] + qxq['zx-xz']
        pq.z = qxq['tz+zt'] + qxq['xy-yx']
                    
        return pq
    
    def invert(self):
        """The inverse of a quaternion."""
        
        q_conj = self.conj()
        q_norm = self.norm()
        
        if q_norm.t == 0:
            print("oops, zero on the norm.")
            return self.q0()
        
        q_norm_inv = Qh([1.0 / q_norm.t, 0, 0, 0])
        q_inv = q_conj.product(q_norm_inv)
        
        return q_inv
    
    def divide_by(self, q1):
        """Divide one quaternion by another. The order matters unless one is using a norm (real number)."""
        
        q1_inv = q1.invert()
        q_div = self.product(q1.invert()) 
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        triple = self.product(q1).product(q2)
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm of one).
    def rotate(self, a_1=0, a_2=0, a_3=0):
        """Do a rotation given up to three angles."""
    
        u = Qh([0, a_1, a_2, a_3])
        u_abs = u.abs_of_q()
        u_normalized = u.divide_by(u_abs)

        q_rot = u_normalized.triple_product(self, u_normalized.conj())
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0):
        """A boost along the x, y, and/or z axis."""
        
        boost = Qh(sr_gamma_betas(beta_x, beta_y, beta_z))      
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Qh([0.5, 0, 0, 0]))
        triple_123 = triple_1.add(half_23)
        
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        exp_g = exp(dimensionless_g)
        
        g_q = Qh()
        g_q.t = self.t / exp_g
        g_q.x = self.x * exp_g
        g_q.y = self.y * exp_g
        g_q.z = self.z * exp_g
        
        return g_q


# Write tests the Qh class

# In[4]:

class TestQh(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Qh([1, -2, -3, -4])
    q2 = Qh([0, 4, -3, 0])
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1.t == 1)
    
    def test_q0(self):
        qz = self.q1.q0()
        if self.verbose: print("q0: {}".format(qz))
        self.assertTrue(qz.t == 0)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_q1(self):
        qz = self.q1.q1()
        if self.verbose: print("q1: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
                
    def test_conj_0(self):
        qz = self.q1.conj()
        if self.verbose: print("q_conj 0: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 2)
        self.assertTrue(qz.y == 3)
        self.assertTrue(qz.z == 4)
                 
    def test_conj_1(self):
        qz = self.q1.conj(1)
        if self.verbose: print("q_conj 1: {}".format(qz))
        self.assertTrue(qz.t == -1)
        self.assertTrue(qz.x == -2)
        self.assertTrue(qz.y == 3)
        self.assertTrue(qz.z == 4)
                 
    def test_conj_2(self):
        qz = self.q1.conj(2)
        if self.verbose: print("q_conj 2: {}".format(qz))
        self.assertTrue(qz.t == -1)
        self.assertTrue(qz.x == 2)
        self.assertTrue(qz.y == -3)
        self.assertTrue(qz.z == 4)
        
    def test_square(self):
        qz = self.q1.square()
        if self.verbose: print("square: {}".format(qz))
        self.assertTrue(qz.t == -28)
        self.assertTrue(qz.x == -4)
        self.assertTrue(qz.y == -6)
        self.assertTrue(qz.z == -8)
                
    def test_norm(self):
        qz = self.q1.norm()
        if self.verbose: print("norm: {}".format(qz))
        self.assertTrue(qz.t == 30)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_norm_of_vector(self):
        qz = self.q1.norm_of_vector()
        if self.verbose: print("norm_of_vector: {}".format(qz))
        self.assertTrue(qz.t == 29)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_abs_of_q(self):
        qz = self.q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(qz))
        self.assertTrue(qz.t == 5)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_abs_of_vector(self):
        qz = self.q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(qz))
        self.assertTrue(qz.t == 5)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_add(self):
        qz = self.q1.add(self.q2)
        if self.verbose: print("add: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 2)
        self.assertTrue(qz.y == -6)
        self.assertTrue(qz.z == -4)
        
    def test_dif(self):
        qz = self.q1.dif(self.q2)
        if self.verbose: print("dif: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == -6)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == -4) 

    def test_product(self):
        qz = self.q1.product(self.q2)
        if self.verbose: print("product: {}".format(qz))
        self.assertTrue(qz.t == -1)
        self.assertTrue(qz.x == -8)
        self.assertTrue(qz.y == -19)
        self.assertTrue(qz.z == 18)
        
    def test_invert(self):
        qz = self.q2.invert()
        if self.verbose: print("invert: {}".format(qz))
        self.assertTrue(qz.t == 0)
        self.assertTrue(qz.x == -0.16)
        self.assertTrue(qz.y == 0.12)
        self.assertTrue(qz.z == 0)
                
    def test_divide_by(self):
        qz = self.q1.divide_by(self.q1)
        if self.verbose: print("divide_by: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0) 
        
    def test_triple_product(self):
        qz = self.q1.triple_product(self.q2, self.q1)
        if self.verbose: print("triple product: {}".format(qz))
        self.assertTrue(qz.t == -2)
        self.assertTrue(qz.x == 124)
        self.assertTrue(qz.y == -84)
        self.assertTrue(qz.z == 8)
        
    def test_rotate(self):
        qz = self.q1.rotate(1)
        if self.verbose: print("rotate: {}".format(qz))
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == -2)
        self.assertTrue(qz.y == 3)
        self.assertTrue(qz.z == 4)
        
    def test_boost(self):
        q1_sq = self.q1.square()
        qz = self.q1.boost(0.003)
        qz2 = qz.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(qz))
        if self.verbose: print("b squared: {}".format(qz2))
        self.assertTrue(round(qz2.t, 12) == round(q1_sq.t, 12))

    def test_g_shift(self):
        q1_sq = self.q1.square()
        qz = self.q1.g_shift(0.003)
        qz2 = qz.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(qz))
        if self.verbose: print("g squared: {}".format(qz2))
        self.assertTrue(qz2.t != q1_sq.t)
        self.assertTrue(qz2.x == q1_sq.x)
        self.assertTrue(qz2.y == q1_sq.y)
        self.assertTrue(qz2.z == q1_sq.z)


# In[5]:

suite = unittest.TestLoader().loadTestsFromModule(TestQh())
unittest.TextTestRunner().run(suite)


# My long term goal is to deal with quaternions on a quaternion manifold. This will have 4 pairs of doublets. Each doublet is paired with its additive inverse. Instead of using real numbers, one uses (3, 0) and (0, 2) to represent +3 and -2 respectively. Numbers such as (5, 6) are allowed. That can be "reduced" to (0, 1).  My sense is that somewhere deep in the depths of relativistic quantum field theory, this will be a "good thing". For now, it is a minor pain to program.

# In[6]:

class Doublet:
    """A pair of number that are additive inverses. It can take
    ints, floats, Symbols, or strings."""
    
    def __init__(self, numbers=None):
        
        if numbers is None:
            self.p = 0
            self.n = 0
            
        elif isinstance(numbers, (int, float)):
            if numbers < 0:
                self.n = -1 * numbers
                self.p = 0
            else:
                self.p = numbers
                self.n = 0
        
        elif isinstance(numbers, Symbol):
            self.p = numbers
            self.n = 0
            
        elif isinstance(numbers, list):
            
            if len(numbers) == 2:
                self.p = numbers[0]
                self.n = numbers[1]

                      
        elif isinstance(numbers, str):
            n_list = numbers.split()
            
            if (len(n_list) == 1):
                if n_list.isnumeric():
                    n_value = float(numbers)
                      
                    if n_value < 0:
                        self.n = -1 * n_list[0]
                        self.p = 0
                      
                    else:
                        self.p = n_list[0]
                        self.n = 0
                        
                else:
                    self.p = Symbol(n_list[0])
                    self.n = 0
                      
            if (len(n_list) == 2):
                if n_list[0].isnumeric():
                    self.p = float(n_list[0])
                else:
                    self.p = Symbol(n_list[0])
                    
                if n_list[1].isnumeric():
                    self.n = float(n_list[1])
                else:
                    self.n = Symbol(n_list[1])
        else:
            print ("unable to parse this Double.")

    def __str__(self):
        """Customize the output."""
        return "{p}p  {n}n".format(p=self.p, n=self.n)
        
    def d_add(self, d1):
        """Add a doublet to another."""
                        
        pa0, n0 = self.p, self.n
        p1, n1 = d1.p, d1.n
                        
        return Doublet([pa0 + p1, n0 + n1])

    def d_flip(self):
        """Flips additive inverses."""
                        
        return Doublet([self.n, self.p])
                        
    def d_dif(self, d1):
        """Take the difference by flipping and adding."""
        d2 = d1.d_flip()
                        
        return self.d_add(d2)
    
    def d_reduce(self):
        """If p and n are not zero, subtract """
        if self.p == 0 or self.n == 0:
            return Doublet([self.p, self.n])
        
        elif self.p > self.n:
            return Doublet([self.p - self.n, 0])
        
        elif self.p < self.n:
            return Doublet([0, self.n - self.p])
        
        else:
            return Doublet()
        
    def Z2_product(self, d1):
        """Uset the Abelian cyclic group Z2 to form the product of 2 doublets."""
        p1 = self.p * d1.p + self.n * d1.n
        n1 = self.p * d1.n + self.n * d1.p
        
        return Doublet([p1, n1])


# In[7]:

class TestDoublet(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    d1 = Doublet()
    d2 = Doublet(2)
    d3 = Doublet(-3)
    dstr12 = Doublet("1 2")
    dstr13 = Doublet("3 2")
    
    def test_null(self):
        self.assertTrue(self.d1.p == 0)
        self.assertTrue(self.d1.n == 0)
       
    def test_2(self):
        self.assertTrue(self.d2.p == 2)
        self.assertTrue(self.d2.n == 0)
        
    def test_3(self):
        self.assertTrue(self.d3.p == 0)
        self.assertTrue(self.d3.n == 3)
    
    def test_str12(self):
        self.assertTrue(self.dstr12.p == 1)
        self.assertTrue(self.dstr12.n == 2)
    
    def test_add(self):
        d_add = self.d2.d_add(self.d3)
        self.assertTrue(d_add.p == 2)
        self.assertTrue(d_add.n == 3)
        
    def test_d_flip(self):
        d_f = self.d2.d_flip()
        self.assertTrue(d_f.p == 0)
        self.assertTrue(d_f.n == 2)
        
    def test_dif(self):
        d_d = self.d2.d_dif(self.d3)
        self.assertTrue(d_d.p == 5)
        self.assertTrue(d_d.n == 0)
            
    def test_reduce(self):
        d_add = self.d2.d_add(self.d3)
        d_r = d_add.d_reduce()
        self.assertTrue(d_r.p == 0)
        self.assertTrue(d_r.n == 1)
        
    def test_Z2_product(self):
        Z2p = self.dstr12.Z2_product(self.dstr13)
        self.assertTrue(Z2p.p == 7)
        self.assertTrue(Z2p.n == 8)
        
    def test_reduced_product(self):
        """Reduce before or after, should make no difference."""
        Z2p_1 = self.dstr12.Z2_product(self.dstr13)
        Z2p_red = Z2p_1.d_reduce()
        d_r_1 = self.dstr12.d_reduce()
        d_r_2 = self.dstr13.d_reduce()
        Z2p_2 = d_r_1.Z2_product(d_r_2)
        self.assertTrue(Z2p_red.p == Z2p_2.p)
        self.assertTrue(Z2p_red.n == Z2p_2.n)


# In[8]:

suite = unittest.TestLoader().loadTestsFromModule(TestDoublet())
unittest.TextTestRunner().run(suite)


# Write a class to handle quaternions given 8 numbers.

# In[9]:

class Qq:
    """Quaternions on a quaternion manifold."""

    def __init__(self, values=None):
        if values is None:
            self.dt, self.dx, self.dy, self.dz = Doublet(), Doublet(),Doublet(), Doublet()
        elif isinstance(values, list):
            if len(values) == 4:
                self.dt = Doublet(values[0])
                self.dx = Doublet(values[1])
                self.dy = Doublet(values[2])
                self.dz = Doublet(values[3])
        
            if len(values) == 8:
                self.dt = Doublet([values[0], values[1]])
                self.dx = Doublet([values[2], values[3]])
                self.dy = Doublet([values[4], values[5]])
                self.dz = Doublet([values[6], values[7]])
                

    def __str__(self):
        """Customize the output."""
        return "{tp}_I0  {tn}_I2  {xp}_i1  {xn}_i3  {yp}_j1  {yn}_j3  {zp}_k1  {zn}_k3".format(tp=self.dt.p, tn=self.dt.n, 
                                                             xp=self.dx.p, xn=self.dx.n, 
                                                             yp=self.dy.p, yn=self.dy.n, 
                                                             zp=self.dz.p, zn=self.dz.n)
        
    def q0(self):
        """Return a zero quaternion."""
        
        return Qq()
      
    def q1(self):
        """Return a multiplicative identity quaternion."""
        
        return Qq([1, 0, 0, 0])
    
    def conj(self, type=0):
        """Three types of conjugates."""
        
        conjq = Qq()

        if type == 0:
            conjq.dt = self.dt
            conjq.dx = self.dx.d_flip()
            conjq.dy = self.dy.d_flip()
            conjq.dz = self.dz.d_flip()
        
        if type == 1:
            conjq.dt = self.dt.d_flip()
            conjq.dx = self.dx
            conjq.dy = self.dy.d_flip()
            conjq.dz = self.dz.d_flip()
        
        if type == 2:
            conjq.dt = self.dt.d_flip()
            conjq.dx = self.dx.d_flip()
            conjq.dy = self.dy
            conjq.dz = self.dz.d_flip()

        return conjq

    def commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt': self.dt.Z2_product(q1.dt),
                    'xx+yy+zz': self.dx.Z2_product(q1.dx).d_add(self.dy.Z2_product(q1.dy)).d_add(self.dz.Z2_product(q1.dz)),
        
                    'tx+xt': self.dt.Z2_product(q1.dx).d_add(self.dx.Z2_product(q1.dt)),
                    'ty+yt': self.dt.Z2_product(q1.dy).d_add(self.dy.Z2_product(q1.dt)),
                    'tz+zt': self.dt.Z2_product(q1.dz).d_add(self.dz.Z2_product(q1.dt))}
        
        return products
    
    def anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        products = {'yz-zy': self.dy.Z2_product(q1.dz).d_dif(self.dz.Z2_product(q1.dy)),
                    'zx-xz': self.dz.Z2_product(q1.dx).d_dif(self.dx.Z2_product(q1.dz)),
                    'xy-yx': self.dx.Z2_product(q1.dy).d_dif(self.dy.Z2_product(q1.dx))}
        
        return products
    
    def all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self.commuting_products(q1)
        products.update(self.anti_commuting_products(q1))
        
        return products
    
    def square(self):
        """Square a quaternion."""
        
        qxq = self.commuting_products(self)
        
        sq_q = Qq()        
        sq_q.dt = qxq['tt'].d_dif(qxq['xx+yy+zz'])
        sq_q.dx = qxq['tx+xt']
        sq_q.dy = qxq['ty+yt']
        sq_q.dz = qxq['tz+zt']

        return sq_q

    
    def reduce(self):
        """Put all doublets into the reduced form so one of each pair is zero."""

        q_red = Qq()
        q_red.dt = self.dt.d_reduce()
        q_red.dx = self.dx.d_reduce()
        q_red.dy = self.dy.d_reduce()
        q_red.dz = self.dz.d_reduce()
        
        return q_red
    
    def norm(self):
        """The norm of a quaternion."""
        
        qxq = self.commuting_products(self)
        
        n_q = Qq()        
        n_q.dt = qxq['tt'].d_add(qxq['xx+yy+zz'])

        return n_q
    
    def norm_of_vector(self):
        """The norm of the vector of a quaternion."""
        
        qxq = self.commuting_products(self)
        
        nv_q = Qq()
        nv_q.dt = qxq['xx+yy+zz']

        return nv_q
    
        
    def abs_of_q(self):
        """The absolute value, the square root of the norm."""

        a = self.norm()
        sqrt_t = a.dt.p ** (1/2)
        a.dt = Doublet(sqrt_t)
        
        return a

    def abs_of_vector(self):
        """The absolute value of the vector, the square root of the norm of the vector."""

        av = self.norm_of_vector()
        sqrt_t = av.dt.p ** (1/2)
        av.dt = Doublet(sqrt_t)
        
        return av
    
    def add(self, q1):
        """Form a add given 2 quaternions."""

        add_q = Qq()
        add_q.dt = self.dt.d_add(q1.dt)
        add_q.dx = self.dx.d_add(q1.dx)
        add_q.dy = self.dy.d_add(q1.dy)
        add_q.dz = self.dz.d_add(q1.dz)
                    
        return add_q    

    def dif(self, q1):
        """Form a add given 2 quaternions."""

        dif_q = Qq()
        dif_q.dt = self.dt.d_dif(q1.dt)
        dif_q.dx = self.dx.d_dif(q1.dx)
        dif_q.dy = self.dy.d_dif(q1.dy)
        dif_q.dz = self.dz.d_dif(q1.dz)
                    
        return dif_q
    
    def product(self, q1):
        """Form a product given 2 quaternions."""

        qxq = self.all_products(q1)
        pq = Qq()
        pq.dt = qxq['tt'].d_dif(qxq['xx+yy+zz'])
        pq.dx = qxq['tx+xt'].d_add(qxq['yz-zy'])
        pq.dy = qxq['ty+yt'].d_add(qxq['zx-xz'])
        pq.dz = qxq['tz+zt'].d_add(qxq['xy-yx'])
                    
        return pq

    def invert(self):
        """Invert a quaternion."""
        
        q_conj = self.conj()
        q_norm = self.norm()
        
        if q_norm.dt.p == 0:
            return self.q0()
        
        q_norm_inv = Qq([1.0 / q_norm.dt.p, 0, 0, 0, 0, 0, 0, 0])
        q_inv = q_conj.product(q_norm_inv)
        
        return q_inv

    def divide_by(self, dq1):
        """Divide one quaternion by another. The order matters unless one is using a norm (real number)."""

        q_inv = dq1.invert()
        q_div = self.product(q_inv) 
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        triple = self.product(q1).product(q2)
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm of one).
    def rotate(self, a_1p=0, a_1n=0, a_2p=0, a_2n=0, a_3p=0, a_3n=0):
        """Do a rotation given up to three angles."""
    
        u = Qq([0, 0, a_1p, a_1n, a_2p, a_2n, a_3p, a_3n])
        u_abs = u.abs_of_q()
        u_normalized = u.divide_by(u_abs)

        q_rot = u_normalized.triple_product(self, u_normalized.conj())
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0):
        """A boost along the x, y, and/or z axis."""
        
        boost = Qq(sr_gamma_betas(beta_x, beta_y, beta_z))
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Qq([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23)
        
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        exp_g = exp(dimensionless_g)
        
        g_q = Qq()
        g_q.dt = Doublet([self.dt.p / exp_g, self.dt.n / exp_g])
        g_q.dx = Doublet([self.dx.p * exp_g, self.dx.n * exp_g])
        g_q.dy = Doublet([self.dy.p * exp_g, self.dy.n * exp_g])
        g_q.dz = Doublet([self.dz.p * exp_g, self.dz.n * exp_g])
        
        return g_q


# In[10]:

class TestQq(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Qq([1, 0, 0, 2, 0, 3, 0, 4])
    q2 = Qq([0, 0, 4, 0, 0, 3, 0, 0])
    q_big = Qq([1, 2, 3, 4, 5, 6, 7, 8])
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1.dt.p == 1)
    
    def test_q0(self):
        qz = self.q1.q0()
        if self.verbose: print("q0: {}".format(qz))
        self.assertTrue(qz.dt.p == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 0)
        
    def test_q1(self):
        qz = self.q1.q1()
        if self.verbose: print("q1: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dz.p == 0)
                
    def test_conj_0(self):
        qz = self.q1.conj()
        if self.verbose: print("conj 0: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dx.p == 2)
        self.assertTrue(qz.dy.p == 3)
        self.assertTrue(qz.dz.p == 4)
                 
    def test_conj_1(self):
        qz = self.q1.conj(1)
        if self.verbose: print("conj 1: {}".format(qz))
        self.assertTrue(qz.dt.n == 1)
        self.assertTrue(qz.dx.n == 2)
        self.assertTrue(qz.dy.p == 3)
        self.assertTrue(qz.dz.p == 4)
                 
    def test_conj_2(self):
        qz = self.q1.conj(2)
        if self.verbose: print("conj 2: {}".format(qz))
        self.assertTrue(qz.dt.n == 1)
        self.assertTrue(qz.dx.p == 2)
        self.assertTrue(qz.dy.n == 3)
        self.assertTrue(qz.dz.p == 4)
        
    def test_square(self):
        q_sq = self.q1.square()
        q_sq_red = q_sq.reduce()
        if self.verbose: print("square: {}".format(q_sq))
        if self.verbose: print("square reduced: {}".format(q_sq_red))
        self.assertTrue(q_sq.dt.p == 1)
        self.assertTrue(q_sq.dt.n == 29)
        self.assertTrue(q_sq.dx.n == 4)
        self.assertTrue(q_sq.dy.n == 6)
        self.assertTrue(q_sq.dz.n == 8)
        self.assertTrue(q_sq_red.dt.p == 0)
        self.assertTrue(q_sq_red.dt.n == 28)
                
    def test_reduce(self):
        q_red = self.q_big.reduce()
        if self.verbose: print("q_big reduced: {}".format(q_red))
        self.assertTrue(q_red.dt.p == 0)
        self.assertTrue(q_red.dt.n == 1)
        self.assertTrue(q_red.dx.p == 0)
        self.assertTrue(q_red.dx.n == 1)
        self.assertTrue(q_red.dy.p == 0)
        self.assertTrue(q_red.dy.n == 1)
        self.assertTrue(q_red.dz.p == 0)
        self.assertTrue(q_red.dz.n == 1)
        
    def test_norm(self):
        qz = self.q1.norm()
        if self.verbose: print("norm: {}".format(qz))
        self.assertTrue(qz.dt.p == 30)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 0)
        
    def test_norm_of_vector(self):
        qz = self.q1.norm_of_vector()
        if self.verbose: print("norm_of_vector: {}".format(qz))
        self.assertTrue(qz.dt.p == 29)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 0)
        
    def test_abs_of_q(self):
        qz = self.q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(qz))
        self.assertTrue(qz.dt.p == 5)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.n == 0)
        
    def test_abs_of_vector(self):
        qz = self.q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(qz))
        self.assertTrue(qz.dt.p == 5)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.n == 0)
        
    def test_add(self):
        qz = self.q1.add(self.q2)
        if self.verbose: print("add: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 4)
        self.assertTrue(qz.dx.n == 2)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 6)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 4)
        
    def test_add_reduce(self):
        qz_red = self.q1.add(self.q2).reduce()
        if self.verbose: print("add reduce: {}".format(qz_red))
        self.assertTrue(qz_red.dt.p == 1)
        self.assertTrue(qz_red.dt.n == 0)
        self.assertTrue(qz_red.dx.p == 2)
        self.assertTrue(qz_red.dx.n == 0)
        self.assertTrue(qz_red.dy.p == 0)
        self.assertTrue(qz_red.dy.n == 6)
        self.assertTrue(qz_red.dz.p == 0)
        self.assertTrue(qz_red.dz.n == 4)
        
    def test_dif(self):
        qz = self.q1.dif(self.q2)
        if self.verbose: print("dif: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 6) 
        self.assertTrue(qz.dy.p == 3)
        self.assertTrue(qz.dy.n == 3)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 4) 

    def test_product(self):
        qz = self.q1.product(self.q2).reduce()
        if self.verbose: print("product: {}".format(qz))
        self.assertTrue(qz.dt.p == 0)
        self.assertTrue(qz.dt.n == 1)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 8)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 19)
        self.assertTrue(qz.dz.p == 18)
        self.assertTrue(qz.dz.n == 0)
        
    def test_invert(self):
        qz = self.q2.invert().reduce()
        if self.verbose: print("inverse: {}".format(qz))
        self.assertTrue(qz.dt.p == 0)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 0.16)
        self.assertTrue(qz.dy.p == 0.12)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 0)

    def test_divide_by(self):
        qz = self.q1.divide_by(self.q1).reduce()
        if self.verbose: print("inverse: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 0)
        self.assertTrue(qz.dz.n == 0) 
        
    def test_triple_product(self):
        qz = self.q1.triple_product(self.q2, self.q1).reduce()
        if self.verbose: print("triple: {}".format(qz))
        self.assertTrue(qz.dt.p == 0)
        self.assertTrue(qz.dt.n == 2)
        self.assertTrue(qz.dx.p == 124)
        self.assertTrue(qz.dx.n == 0)
        self.assertTrue(qz.dy.p == 0)
        self.assertTrue(qz.dy.n == 84)
        self.assertTrue(qz.dz.p == 8)
        self.assertTrue(qz.dz.n == 0)
        
    def test_rotate(self):
        qz = self.q1.rotate(1).reduce()
        if self.verbose: print("rotate: {}".format(qz))
        self.assertTrue(qz.dt.p == 1)
        self.assertTrue(qz.dt.n == 0)
        self.assertTrue(qz.dx.p == 0)
        self.assertTrue(qz.dx.n == 2)
        self.assertTrue(qz.dy.p == 3)
        self.assertTrue(qz.dy.n == 0)
        self.assertTrue(qz.dz.p == 4)
        self.assertTrue(qz.dz.n == 0)
        
    def test_boost(self):
        q1_sq = self.q1.square().reduce()
        qz = self.q1.boost(0.003)
        qz2 = qz.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(qz))
        if self.verbose: print("b squared: {}".format(qz2))
        self.assertTrue(round(qz2.dt.n, 12) == round(q1_sq.dt.n, 12))
        
    def test_g_shift(self):
        q1_sq = self.q1.square().reduce()
        qz = self.q1.g_shift(0.003)
        qz2 = qz.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(qz))
        if self.verbose: print("g squared: {}".format(qz2))
        self.assertTrue(qz2.dt.n != q1_sq.dt.n)
        self.assertTrue(qz2.dx.p == q1_sq.dx.p)
        self.assertTrue(qz2.dx.n == q1_sq.dx.n)
        self.assertTrue(qz2.dy.p == q1_sq.dy.p)
        self.assertTrue(qz2.dy.n == q1_sq.dy.n)
        self.assertTrue(qz2.dz.p == q1_sq.dz.p)
        self.assertTrue(qz2.dz.n == q1_sq.dz.n)


# In[11]:

suite = unittest.TestLoader().loadTestsFromModule(TestQq())
unittest.TextTestRunner().run(suite)


# Create a class that can figure out if two quaternions are in the same equivalence class. An equivalence class of space-time is a subset of events in space-time. For 
# 
