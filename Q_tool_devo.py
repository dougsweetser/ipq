
# coding: utf-8

# # Developing Quaternion Tools for iPython3

# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.

# In[1]:

def gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""
    
    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** 0.5

def gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""
    
    g = gamma(beta_x, beta_y, beta_z)
    
    return [g, g * beta_x, g * beta_y, g * beta_z]


# Define a class Qh to manipulate quaternions as Hamilton would have done it so many years ago. Note: I do think we can learn more of the silent workings of Nature using a quaternion on a quaternion manifold, but for now this is much easier to implement.

# In[2]:

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
    
    def square(self):
        """Square a quaternion."""
        t, x, y, z = self.t, self.x, self.y, self.z
        
        sq_q = Qh()
        sq_q.t = t * t - x * x - y * y - z * z
        sq_q.x = 2 * t * x
        sq_q.y = 2 * t * y
        sq_q.z = 2 * t * z

        return sq_q
    
    def norm(self):
        """The norm of a quaternion."""
        t, x, y, z = self.t, self.x, self.y, self.z
        
        n_q = Qh()
        n_q.t = t * t + x * x + y * y + z * z
        n_q.x = 0
        n_q.y = 0
        n_q.z = 0

        return n_q
    
    def norm_of_vector(self):
        """The norm of the vector of a quaternion."""
        x, y, z = self.x, self.y, self.z
        
        nv_q = Qh()
        nv_q.t = x * x + y * y + z * z
        nv_q.x = 0
        nv_q.y = 0
        nv_q.z = 0

        return nv_q
    
    def abs(self):
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
        
    def sum(self, q1):
        """Form a sum given 2 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        
        sum_q = Qh()
        sum_q.t = t1 + t2
        sum_q.x = x1 + x2
        sum_q.y = y1 + y2
        sum_q.z = z1 + z2
                    
        return sum_q    

    def dif(self, q1):
        """Form a sum given 2 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        
        sum_q = Qh()
        sum_q.t = t1 - t2
        sum_q.x = x1 - x2
        sum_q.y = y1 - y2
        sum_q.z = z1 - z2
                    
        return sum_q
    
    def invert(self):
        """The inverse of a quaternion."""
        
        q_conj = self.conj()
        q_norm = self.norm()
        
        print("q_conj: {}".format(q_conj))
        print("q_norm: {}".format(q_norm))
        if q_norm.t == 0:
            print("oops, zero on the norm.")
            return self.q0()
        
        q_norm_inv = 1.0 / q_norm.t
        q_inv = q_conj.scalar_product(q_norm_inv)
        
        print("q_norm_inv: {}".format(q_norm_inv))
        print("q_inv: {}".format(q_inv))
        
        return q_inv
    
    def product(self, q1):
        """Form a product given 2 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        
        pq = Qh()
        pq.t = t1 * t2 - x1 * x2 - y1 * y2 - z1 * z2
        pq.x = t2 * x1 + t1 * x2 - y2 * z1 + y1 * z2
        pq.y = t2 * y1 + t1 * y2 + x2 * z1 - x1 * z2
        pq.z = -x2 * y1 + x1 * y2 + t2 * z1 + t1 * z2
                    
        return pq
    
        
    def divide_by(self, q1):
        """Divide one quaternion by another. The order matters unless one is using a norm (real number)."""

        q_div = self.product(q1.invert()) 
        return q_div
    
    def scalar_product(self, scalar):
        """A scalar value times a quaternion."""
        
        t, x, y, z = self.t, self.x, self.y, self.z
        
        scalar_q = Qh()
        scalar_q.t = scalar * t
        scalar_q.x = scalar * x
        scalar_q.y = scalar * y
        scalar_q.z = scalar * z
        
        return scalar_q
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""

        t1, x1, y1, z1 = self.t, self.x, self.y, self.z
        t2, x2, y2, z2 = q1.t, q1.x, q1.y, q1.z
        t3, x3, y3, z3 = q2.t, q2.x, q2.y, q2.z
        
        triple = Qh()
        
        triple.t = t1 * t2 * t3 - t3 * x1 * x2 - t2 * x1 * x3 - t1 * x2 * x3 - t3 * y1 * y2 -                   t2 * y1 * y3 - t1 * y2 * y3 + x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 +                   x1 * y3 * z2 - t3 * z1 * z2 + x2 * y1 * z3 - x1 * y2 * z3 - t2 * z1 * z3 -                   t1 * z2 * z3

        triple.x = t2 * t3 * x1 + t1 * t3 * x2 + t1 * t2 * x3 - x1 * x2 * x3 - x3 * y1 * y2 +                   x2 * y1 * y3 - x1 * y2 * y3 - t3 * y2 * z1 - t2 * y3 * z1 + t3 * y1 * z2 -                   t1 * y3 * z2 - x3 * z1 * z2 + t2 * y1 * z3 + t1 * y2 * z3 + x2 * z1 * z3 -                   x1 * z2 * z3
    
        triple.y = t2 * t3 * y1 - x2 * x3 * y1 + t1 * t3 * y2 + x1 * x3 * y2 + t1 * t2 * y3 -                   x1 * x2 * y3 - y1 * y2 * y3 + t3 * x2 * z1 + t2 * x3 * z1 - t3 * x1 * z2 +                   t1 * x3 * z2 - y3 * z1 * z2 - t2 * x1 * z3 - t1 * x2 * z3 + y2 * z1 * z3 -                   y1 * z2 * z3
    
        triple.z = -t3 * x2 * y1 - t2 * x3 * y1 + t3 * x1 * y2 - t1 * x3 * y2 + t2 * x1 * y3 +                    t1 * x2 * y3 + t2 * t3 * z1 - x2 * x3 * z1 - y2 * y3 * z1 + t1 * t3 * z2 +                    x1 * x3 * z2 + y1 * y3 * z2 + t1 * t2 * z3 - x1 * x2 * z3 - y1 * y2 * z3 -                    z1 * z2 * z3
                    
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm of one).
    def rotate(self, a_1=0, a_2=0, a_3=0):
        """Do a rotation given up to three angles."""
    
        u = Qh([0, a_1, a_2, a_3])
        u_abs = u.abs()
        u_normalized = u.divide_by(u_abs)

        q_rot = u_normalized.triple_product(self, u_normalized.conj())
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0):
        """A boost along the x, y, and/or z axis."""
        
        boost = Qh(gamma_betas(beta_x, beta_y, beta_z))       
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.scalar_product(0.5)
        triple_123 = triple_1.sum(half_23)
        
        return triple_123


# Write tests the Qh class

# In[3]:

import unittest

class TestQh(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Qh([1, 2, 3, 4])
    q2 = Qh([0, 4, 3, 0])
    
    def test_qt(self):
        self.assertTrue(self.q1.t == 1)
    
    def test_q0(self):
        qz = self.q1.q0()
        self.assertTrue(qz.t == 0)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_q1(self):
        qz = self.q1.q1()
        print(qz)
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
                
    def test_conj_0(self):
        qz = self.q1.conj()
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == -2)
        self.assertTrue(qz.y == -3)
        self.assertTrue(qz.z == -4)
                 
    def test_conj_1(self):
        qz = self.q1.conj(1)
        print(qz)
        self.assertTrue(qz.t == -1)
        self.assertTrue(qz.x == 2)
        self.assertTrue(qz.y == -3)
        self.assertTrue(qz.z == -4)
                 
    def test_conj_0(self):
        qz = self.q1.conj(2)
        print(qz)
        self.assertTrue(qz.t == -1)
        self.assertTrue(qz.x == -2)
        self.assertTrue(qz.y == 3)
        self.assertTrue(qz.z == -4)
        
    def test_square(self):
        qz = self.q1.square()
        print(qz)
        self.assertTrue(qz.t == -28)
        self.assertTrue(qz.x == 4)
        self.assertTrue(qz.y == 6)
        self.assertTrue(qz.z == 8)
                
    def test_norm(self):
        qz = self.q1.norm()
        print(qz)
        self.assertTrue(qz.t == 30)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_norm_of_vector(self):
        qz = self.q1.norm_of_vector()
        print(qz)
        self.assertTrue(qz.t == 29)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_abs(self):
        qz = self.q2.abs()
        print(qz)
        self.assertTrue(qz.t == 5)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_abs_of_vector(self):
        qz = self.q2.abs_of_vector()
        print(qz)
        self.assertTrue(qz.t == 5)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0)
        
    def test_sum(self):
        qz = self.q1.sum(self.q2)
        print(qz)
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 6)
        self.assertTrue(qz.y == 6)
        self.assertTrue(qz.z == 4)
        
    def test_dif(self):
        qz = self.q1.dif(self.q2)
        print(qz)
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == -2)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 4) 
        
    def test_invert(self):
        qz = self.q2.invert()
        print(qz)
        self.assertTrue(qz.t == 0)
        self.assertTrue(qz.x == -0.16)
        self.assertTrue(qz.y == -0.12)
        self.assertTrue(qz.z == 0)
        
    def test_product(self):
        qz = self.q1.product(self.q2)
        print(qz)
        self.assertTrue(qz.t == -17)
        self.assertTrue(qz.x == -8)
        self.assertTrue(qz.y == 19)
        self.assertTrue(qz.z == -6)
        
    def test_divide_by(self):
        qz = self.q1.divide_by(self.q1)
        print(qz)
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 0)
        self.assertTrue(qz.y == 0)
        self.assertTrue(qz.z == 0) 

    def test_scalar_product(self):
        qz = self.q1.scalar_product(2)
        print(qz)
        self.assertTrue(qz.t == 2)
        self.assertTrue(qz.x == 4)
        self.assertTrue(qz.y == 6)
        self.assertTrue(qz.z == 8) 
        
    def test_triple_product(self):
        qz = self.q1.triple_product(self.q2, self.q1)
        print(qz)
        self.assertTrue(qz.t == -34)
        self.assertTrue(qz.x == 52)
        self.assertTrue(qz.y == -12)
        self.assertTrue(qz.z == -136)
        
    def test_rotate(self):
        qz = self.q1.rotate(1)
        print(qz)
        self.assertTrue(qz.t == 1)
        self.assertTrue(qz.x == 2)
        self.assertTrue(qz.y == -3)
        self.assertTrue(qz.z == -4)
        
    def test_boost(self):
        qz = self.q1.boost(0.1)
        qz2 = qz.square()
        self.assertTrue(qz2.t == -28)


# In[4]:

import nose

try:
    nose.runmodule(argv=['/Users/doug/Google Drive/IPython_notebooks/Q_tool_devo.ipynb'])
    
except:
    print("ok")


# In[5]:

print(Qh(gamma_betas(0.9999999999999)))


# In[6]:

q1 = Qh([3,2,4,3])
q2 = Qh([422, 2, 55, 553])
q3 = Qh([-2, -12, 5, 0.003])


# In[7]:

print(q1.conj())


# In[8]:

print(q1.dif(q1))
print(q1.sum(q1))
print(q1.product(q1))
print(q1.square())


# In[9]:

q1.

