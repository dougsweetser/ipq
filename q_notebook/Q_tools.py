
# coding: utf-8

# # Developing Quaternion and Space-time Number Tools for iPython3

# In this notebook, tools for working with quaternions for physics issues are developed. The class QH treat quaternions as Hamilton would have done: as a 4-vector over the real numbers. 
# 
# In physics, group theory plays a central role in the fundamental forces of Nature via the standard model. The gauge symmetry U(1) a unit circle in the complex plane leads to electric charge conservation. The unit quaternions $SU(2)$ is the symmetry needed for the weak force which leads to beta decay. The group $SU(3)$ is the symmetry of the strong force that keeps a nucleus together.
# 
# The class Q8 was written in the hope that group theory would be written in first, not added as needed later. I call these "space-time numbers". The problem with such an approach is that one does not use the mathematical field of real numbers. Instead one relies on the set of positive reals. In some ways, this is like reverse engineering some basic computer science. Libraries written in C have a notion of a signed versus unsigned integer. The signed integer behaves like the familiar integers. The unsigned integer is like the positive integers. The difference between the two is whether there is a placeholder for the sign or not. All floats are signed. The modulo operations that work for unsigned integers does not work for floats.
# 
# This set of tools is done 4x:
# 1. QH - Quaternions for Hamilton, can do symbolic manipulations
# 1. QHa - Quaternions for Hamilton numpy arrays
# 1. Q8 - Quaternions that are represented by 8 numbers
# 1. Q8a - Quaternions that are represented by 8 numbers that are numpy arrays
# 
# Test driven development was used. The same tests were used for QH, QHa, Q8, and Q8a.  Either class can be used to study quaternions in physics.

# In[1]:


import IPython
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import sympy as sp
import os
import unittest

from IPython.display import display
from os.path import basename
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.

# In[2]:


def sr_gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""

    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** 0.5

def sr_gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""

    g = sr_gamma(beta_x, beta_y, beta_z)
    
    return [g, g * beta_x, g * beta_y, g * beta_z]


# ## Quaternions for Hamilton

# Define a class QH to manipulate quaternions as Hamilton would have done it so many years ago. The "qtype" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.

# In[3]:


class QH(object):
    """Quaternions as Hamilton would have defined them, on the manifold R^4."""

    def __init__(self, values=None, qtype="Q"):
        if values is None:
            self.t, self.x, self.y, self.z = 0, 0, 0, 0
        elif len(values) == 4:
            self.t, self.x, self.y, self.z = values[0], values[1], values[2], values[3]

        elif len(values) == 8:
            self.t, self.x = values[0] - values[1], values[2] - values[3]
            self.y, self.z = values[4] - values[5], values[6] - values[7]
        self.qtype = qtype

    def __str__(self):
        """Customize the output."""
        return "({t}, {x}, {y}, {z}) {qt}".format(t=self.t, x=self.x, y=self.y, z=self.z, qt=self.qtype)
    
    def display_q(self):
        """display each terms in a pretty way."""

        display((self.t, self.x, self.y, self.z, self.qtype))
        return

    
    def simple_q(self):
        """display each terms in a pretty way."""
        
        self.t = sp.simplify(self.t)
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)
        self.z = sp.simplify(self.z)
        return
    
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""

        return QH([0, 0, 0, 0], qtype=qtype)

    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""

        return QH([1, 0, 0, 0], qtype=qtype)
    
    def q_i(self, qtype="i"):
        """Return i."""

        return QH([0, 1, 0, 0], qtype=qtype)
    
    def q_j(self, qtype="j"):
        """Return j."""

        return QH([0, 0, 1, 0], qtype=qtype)
    
    def q_k(self, qtype="k"):
        """Return k."""

        return QH([0, 0, 0, 1], qtype=qtype)

    def dupe(self, qtype=""):
        """Return a duplicate copy, good for testing since qtypes persist"""
        
        return QH([self.t, self.x, self.y, self.z], qtype=self.qtype)
    
    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        self_t, self_x, self_y, self_z = sp.expand(self.t), sp.expand(self.x), sp.expand(self.y), sp.expand(self.z)
        q2_t, q2_x, q2_y, q2_z = sp.expand(q2.t), sp.expand(q2.x), sp.expand(q2.y), sp.expand(q2.z)
        
        if (self_t == q2_t) and (self_x == q2_x) and (self_y == q2_y) and (self_z == q2_z):
            return True
        
        else:
            return False
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""

        t, x, y, z = self.t, self.x, self.y, self.z
        conj_q = QH(qtype=self.qtype)

        if conj_type == 0:
            conj_q.t = t
            conj_q.x = -1 * x
            conj_q.y = -1 * y
            conj_q.z = -1 * z

        if conj_type == 1:
            conj_q.t = -1 * t
            conj_q.x = x
            conj_q.y = -1 * y
            conj_q.z = -1 * z
            qtype += "1"
            
        if conj_type == 2:
            conj_q.t = -1 * t
            conj_q.x = -1 * x
            conj_q.y = y
            conj_q.z = -1 * z
            qtype += "2"
            
        conj_q.add_qtype(qtype)
        return conj_q
    
    def flip_signs(self, qtype="*-1"):
        """Flip the signs of all terms."""
        
        t, x, y, z = self.t, self.x, self.y, self.z
        flip_q = QH(qtype=self.qtype)

        flip_q.t = -1 * t
        flip_q.x = -1 * x
        flip_q.y = -1 * y
        flip_q.z = -1 * z
        
        flip_q.add_qtype(qtype)
        return flip_q

    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""

        t, x, y, z = self.t, self.x, self.y, self.z
        conj_q = QH(qtype=self.qtype)

        if conj_type == '-':
            conj_q.t = t
            conj_q.x = -1 * x
            conj_q.y = -1 * y
            conj_q.z = -1 * z
            qtype += "*-"

        if conj_type == "'":
            conj_q.t = t
            conj_q.x = -1 * x
            conj_q.y = -1 * y
            conj_q.z = z
            qtype += "*'"
            
        if conj_type == '*':
            conj_q.t = t
            conj_q.x = x
            conj_q.y = y
            conj_q.z = -1 * z
            qtype += "*"
            
        conj_q.add_qtype(qtype)
        return conj_q

    def add_qtype(self, qtype):
        """Adds a qtype to an exiting qtype."""
        
        self.qtype += "." + qtype
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        s_t, s_x, s_y, s_z = self.t, self.x, self.y, self.z
        q1_t, q1_x, q1_y, q1_z = q1.t, q1.x, q1.y, q1.z

        products = {'tt': s_t * q1_t,
                    'xx+yy+zz': s_x * q1_x + s_y * q1_y + s_z * q1_z,
                    'tx+xt': s_t * q1_x + s_x * q1_t,
                    'ty+yt': s_t * q1_y + s_y * q1_t,
                    'tz+zt': s_t * q1_z + s_z * q1_t}

        return products

    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        s_x, s_y, s_z = self.x, self.y, self.z
        q1_x, q1_y, q1_z = q1.x, q1.y, q1.z

        products = {'yz-zy': s_y * q1_z - s_z * q1_y,
                    'zx-xz': s_z * q1_x - s_x * q1_z,
                    'xy-yx': s_x * q1_y - s_y * q1_x
                    }

        return products

    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))

        return products

    def square(self, qtype="sq"):
        """Square a quaternion."""

        qxq = self._commuting_products(self)

        sq_q = QH(qtype=self.qtype)
        sq_q.t = qxq['tt'] - qxq['xx+yy+zz']
        sq_q.x = qxq['tx+xt']
        sq_q.y = qxq['ty+yt']
        sq_q.z = qxq['tz+zt']

        sq_q.add_qtype(qtype)
        return sq_q

    def norm_squared(self, qtype="norm_squared"):
        """The norm_squared of a quaternion."""

        qxq = self._commuting_products(self)

        n_q = QH(qtype=self.qtype)
        n_q.t = qxq['tt'] + qxq['xx+yy+zz']

        n_q.add_qtype(qtype)
        return n_q

    def norm_squared_of_vector(self, qtype="norm_squaredV"):
        """The norm_squared of the vector of a quaternion."""

        qxq = self._commuting_products(self)

        nv_q = QH(qtype=self.qtype)
        nv_q.t = qxq['xx+yy+zz']

        nv_q.add_qtype(qtype)
        return nv_q

    def abs_of_q(self, qtype="abs"):
        """The absolute value, the square root of the norm_squared."""

        a = self.norm_squared()
        sqrt_t = a.t ** 0.5
        a.t = sqrt_t

        a.qtype = self.qtype
        a.add_qtype(qtype)
        return a

    def abs_of_vector(self, qtype="absV"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        av = self.norm_squared_of_vector()
        sqrt_t = av.t ** 0.5
        av.t = sqrt_t

        av.qtype = self.qtype
        av.add_qtype(qtype)
        return av

    def add(self, qh_1, qtype=""):
        """Form a add given 2 quaternions."""

        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z
        t_2, x_2, y_2, z_2 = qh_1.t, qh_1.x, qh_1.y, qh_1.z

        add_q = QH()
        add_q.t = t_1 + t_2
        add_q.x = x_1 + x_2
        add_q.y = y_1 + y_2
        add_q.z = z_1 + z_2
        
        if qtype:
            add_q.qtype = qtype
        else:
            add_q.qtype = "{f}+{s}".format(f=self.qtype, s=qh_1.qtype)
        
        return add_q    

    def dif(self, qh_1, qtype=""):
        """Form a add given 2 quaternions."""

        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z
        t_2, x_2, y_2, z_2 = qh_1.t, qh_1.x, qh_1.y, qh_1.z

        dif_q = QH()
        dif_q.t = t_1 - t_2
        dif_q.x = x_1 - x_2
        dif_q.y = y_1 - y_2
        dif_q.z = z_1 - z_2

        if qtype:
            dif_q.qtype = qtype
        else:
            dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=qh_1.qtype)
            
        return dif_q

    def product(self, q1, kind="", qtype=""):
        """Form a product given 2 quaternions. Kind can be '' aka standard, even, odd, or even_minus_odd."""
        
        commuting = self._commuting_products(q1)
        q_even = QH()
        q_even.t = commuting['tt'] - commuting['xx+yy+zz']
        q_even.x = commuting['tx+xt']
        q_even.y = commuting['ty+yt']
        q_even.z = commuting['tz+zt']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = QH()
        q_odd.t = 0
        q_odd.x = anti_commuting['yz-zy']
        q_odd.y = anti_commuting['zx-xz']
        q_odd.z = anti_commuting['xy-yx']
        
        result = QH()
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'.")
            
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
            
        return result

    def Euclidean_product(self, q1, kind="", qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        pq = QH()
        pq = self.conj().product(q1, kind)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """The inverse of a quaternion."""

        q_conj = self.conj()
        q_norm_squared = self.norm_squared()

        if q_norm_squared.t == 0:
            print("oops, zero on the norm_squared.")
            return self.q0()

        q_norm_squared_inv = QH([1.0 / q_norm_squared.t, 0, 0, 0])
        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        
        q_inv.add_qtype(qtype)
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""
        
        q1_inv = q1.invert()
        q_div = self.product(q1.invert())
        
        if qtype:
            q_div.qtype = qtype
        else:
            q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
            
        return q_div

    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""

        triple = self.product(q1).product(q2)
        return triple

    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1=0, a_2=0, a_3=0, qtype="rot"):
        """Do a rotation given up to three angles."""

        u = QH([0, a_1, a_2, a_3])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
  
        q_rot.qtype = self.qtype
        q_rot.add_qtype(qtype)
        return q_rot

    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a))
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""

        boost = QH(sr_gamma_betas(beta_x, beta_y, beta_z))      
        b_conj = boost.conj()

        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
      
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(QH([0.5, 0, 0, 0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        
        triple_123.add_qtype(qtype)
        return triple_123

    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    # g_form is the form of the function, either minimal or exponential
    # Minimal is what is needed to pass all weak field tests of gravity
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""

        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self

        g_q = QH(qtype=self.qtype)
        g_q.t = self.t / g_factor
        g_q.x = self.x * g_factor
        g_q.y = self.y * g_factor
        g_q.z = self.z * g_factor

        g_q.add_qtype(qtype)
        return g_q
    
    def q_sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""

        q_out = QH(qtype=self.qtype)
        v_norm = self.norm_squared_of_vector()
        
        if v_norm.t == 0:
            
            q_out.t = sp.sin(self.t)
            q_out.x = 0
            q_out.y = 0
            q_out.z = 0
            
        else:
            
            v_factor = sp.cos(self.t) * sp.sinh(v_norm.t) / v_norm.t
            
            q_out.t = math.trunc(sp.sin(self.t) * sp.cosh(v_norm.t))
            q_out.x = math.trunc(v_factor * self.x)
            q_out.y = math.trunc(v_factor * self.y)
            q_out.z = math.trunc(v_factor * self.z)

        q_out.add_qtype(qtype)
        return q_out
        


# Write tests the QH class.

# In[4]:


class TestQH(unittest.TestCase):
    """Class to make sure all the functions work as expected."""

    Q = QH([1, -2, -3, -4], qtype="Q")
    P = QH([0, 4, -3, 0], qtype="P")
    verbose = True

    def test_qt(self):
        q1 = self.Q.dupe()
        self.assertTrue(q1.t == 1)

    def test_q_0(self):
        q1 = self.Q.dupe()
        q_z = q1.q_0()
        if self.verbose: print("q_0: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_1(self):
        q1 = self.Q.dupe()
        q_z = q1.q_1()
        if self.verbose: print("q_1: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_i(self):
        q1 = self.Q.dupe()
        q_z = q1.q_i()
        if self.verbose: print("q_i: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 1)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_j(self):
        q1 = self.Q.dupe()
        q_z = q1.q_j()
        if self.verbose: print("q_j: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 1)
        self.assertTrue(q_z.z == 0)

    def test_q_k(self):
        q1 = self.Q.dupe()
        q_z = q1.q_k()
        if self.verbose: print("q_k: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 1)
        
    def test_equals(self):
        self.assertTrue(self.Q.equals(self.Q))
        self.assertFalse(self.Q.equals(self.P))

    def test_conj_0(self):
        q1 = self.Q.dupe()
        q_z = q1.conj()
        if self.verbose: print("q_conj 0: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_conj_1(self):
        q1 = self.Q.dupe()
        q_z = q1.conj(1)
        if self.verbose: print("q_conj 1: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_conj_2(self):
        q1 = self.Q.dupe()
        q_z = q1.conj(2)
        if self.verbose: print("q_conj 2: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 4)
        
    def sign_flips(self):
        q1 = self.Q.dupe()
        q_z = q1.sign_flips()
        if self.verbose: print("sign_flips: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)
    
    def test_vahlen_conj_minus(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj()
        if self.verbose: print("q_vahlen_conj -: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_vahlen_conj_star(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj('*')
        if self.verbose: print("q_vahlen_conj *: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 4)

    def test_vahlen_conj_prime(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj("'")
        if self.verbose: print("q_vahlen_conj ': {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == -4)

    def test_square(self):
        q1 = self.Q.dupe()
        q_z = q1.square()
        if self.verbose: print("square: {}".format(q_z))
        self.assertTrue(q_z.t == -28)
        self.assertTrue(q_z.x == -4)
        self.assertTrue(q_z.y == -6)
        self.assertTrue(q_z.z == -8)

    def test_norm_squared(self):
        q1 = self.Q.dupe()
        q_z = q1.norm_squared()
        if self.verbose: print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z.t == 30)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_norm_squared_of_vector(self):
        q1 = self.Q.dupe()
        q_z = q1.norm_squared_of_vector()
        if self.verbose: print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z.t == 29)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_abs_of_q(self):
        q2 = self.P.dupe()
        q_z = q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z.t == 5)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_abs_of_vector(self):
        q2 = self.P.dupe()
        q_z = q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z.t == 5)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_add(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.add(q2)
        if self.verbose: print("add: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == -6)
        self.assertTrue(q_z.z == -4)
        
    def test_dif(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.dif(q2)
        if self.verbose: print("dif: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -6)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == -4) 

    def test_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2)
        if self.verbose: print("product: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == -8)
        self.assertTrue(q_z.y == -19)
        self.assertTrue(q_z.z == 18)
        
    def test_product_even(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="even")
        if self.verbose: print("product, kind even: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 4)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 0)
        
    def test_product_odd(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="odd")
        if self.verbose: print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == -12)
        self.assertTrue(q_z.y == -16)
        self.assertTrue(q_z.z == 18)
        
    def test_product_even_minus_odd(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="even_minus_odd")
        if self.verbose: print("product, kind even_minus_odd: {}".format(q_z))
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 16)
        self.assertTrue(q_z.y == 13)
        self.assertTrue(q_z.z == -18)
        
    def test_Euclidean_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.Euclidean_product(q2)
        if self.verbose: print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 16)
        self.assertTrue(q_z.y == 13)
        self.assertTrue(q_z.z == -18)
        
    def test_invert(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q2.invert()
        if self.verbose: print("invert: {}".format(q_z))
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == -0.16)
        self.assertTrue(q_z.y == 0.12)
        self.assertTrue(q_z.z == 0)
                
    def test_divide_by(self):
        q1 = self.Q.dupe()
        q_z = q1.divide_by(q1)
        if self.verbose: print("divide_by: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0) 
        
    def test_triple_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.triple_product(q2, q1)
        if self.verbose: print("triple product: {}".format(q_z))
        self.assertTrue(q_z.t == -2)
        self.assertTrue(q_z.x == 124)
        self.assertTrue(q_z.y == -84)
        self.assertTrue(q_z.z == 8)
        
    def test_rotate(self):
        q1 = self.Q.dupe()
        q_z = q1.rotate(1)
        if self.verbose: print("rotate: {}".format(q_z))
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)
        
    def test_boost(self):
        q1 = self.Q.dupe()
        q1_sq = q1.square()
        q_z = q1.boost(0.003)
        q_z2 = q_z.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(q_z))
        if self.verbose: print("boosted squared: {}".format(q_z2))
        print("{}")
        self.assertTrue(round(q_z2.t, 5) == round(q1_sq.t, 5))

    def test_g_shift(self):
        q1 = self.Q.dupe()
        q1_sq = q1.square()
        q_z = q1.g_shift(0.003)
        q_z2 = q_z.square()
        q_z_minimal = q1.g_shift(0.003, g_form="minimal")
        q_z2_minimal = q_z_minimal.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(q_z))
        if self.verbose: print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.t != q1_sq.t)
        self.assertTrue(q_z2.x == q1_sq.x)
        self.assertTrue(q_z2.y == q1_sq.y)
        self.assertTrue(q_z2.z == q1_sq.z)
        self.assertTrue(q_z2_minimal.t != q1_sq.t)
        self.assertTrue(q_z2_minimal.x == q1_sq.x)
        self.assertTrue(q_z2_minimal.y == q1_sq.y)
        self.assertTrue(q_z2_minimal.z == q1_sq.z)
        
    def test_q_sin(self):
        QH([0, 0, 0, 0]).q_sin()


# In[5]:


suite = unittest.TestLoader().loadTestsFromModule(TestQH())
unittest.TextTestRunner().run(suite);


# ## Numpy Arrays for Hamilton

# A separate class is needed for numpy array due to technical issues I have getting sympy and numpy to play nicely with each other...

# In[6]:


class QHa(object):
    """Quaternions as nparrays."""

    def __init__(self, values=None, qtype="Q"):
        if values is None:
            self.a = np.array([0.0, 0.0, 0.0, 0.0])
        elif len(values) == 4:
            self.a = np.array([values[0], values[1], values[2], values[3]])

        elif len(values) == 8:
            self.a = np.array([values[0] - values[1], values[2] - values[3], values[4] - values[5], values[6] - values[7]])
        
        self.qtype = qtype

    def __str__(self):
        """Customize the output."""
        return "({t}, {x}, {y}, {z}) {qt}".format(t=self.a[0], x=self.a[1], y=self.a[2], z=self.a[3], qt=self.qtype)
    
    def display_q(self):
        """display each terms in a pretty way."""

        display((self.a[0], self.a[1], self.a[2], self.a[3], self.qtype))
        return

    def simple_q(self):
        """display each terms in a pretty way."""
        
        self.a[0] = sp.simplify(self.a[0])
        self.a[1] = sp.simplify(self.a[1])
        self.a[2] = sp.simplify(self.a[2])
        self.a[3] = sp.simplify(self.a[3])
        return
    
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""

        return QHa([0.0, 0.0, 0.0, 0.0], qtype=qtype)

    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""

        return QHa([1.0, 0.0, 0.0, 0.0], qtype=qtype)
    
    def q_i(self, qtype="i"):
        """Return i."""

        return QHa([0.0, 1.0, 0.0, 0.0], qtype=qtype)

    def q_j(self, qtype="j"):
        """Return j."""

        return QHa([0.0, 0.0, 1.0, 0.0], qtype=qtype)
    
    def q_k(self, qtype="k"):
        """Return k."""

        return QHa([0.0, 0.0, 0.0, 1.0], qtype=qtype)

    def dupe(self, qtype=""):
        """Return a duplicate copy, good for testing since qtypes persist"""
        
        return QHa([self.a[0], self.a[1], self.a[2], self.a[3]], qtype=self.qtype)
    
    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        self_t, self_x, self_y, self_z = sp.expand(self.a[0]), sp.expand(self.a[1]), sp.expand(self.a[2]), sp.expand(self.a[3])
        q2_t, q2_x, q2_y, q2_z = sp.expand(q2.a[0]), sp.expand(q2.a[1]), sp.expand(q2.a[2]), sp.expand(q2.a[3])
        
        if (self_t == q2_t) and (self_x == q2_x) and (self_y == q2_y) and (self_z == q2_z):
            return True
        
        else:
            return False
        
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""

        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        conj_q = QHa(qtype=self.qtype)

        if conj_type == 0:
            conj_q.a[0] = 1.0 * t
            conj_q.a[1] = -1.0 * x
            conj_q.a[2] = -1.0 * y
            conj_q.a[3] = -1.0 * z

        if conj_type == 1:
            conj_q.a[0] = -1.0 * t
            conj_q.a[1] = 1.0 * x
            conj_q.a[2] = -1.0 * y
            conj_q.a[3] = -1.0 * z
            qtype += "1"
            
        if conj_type == 2:
            conj_q.a[0] = -1 * t
            conj_q.a[1] = -1 * x
            conj_q.a[2] = 1.0 * y
            conj_q.a[3] = -1 * z
            qtype += "2"
            
        conj_q.add_qtype(qtype)
        return conj_q
    
    def flip_signs(self, conj_type=0, qtype="*"):
        """Flip all the signs, just like multipying by -1."""

        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        flip_q = QHa(qtype=self.qtype)

        flip_q.a[0] = -1.0 * t
        flip_q.a[1] = -1.0 * x
        flip_q.a[2] = -1.0 * y
        flip_q.a[3] = -1.0 * z
            
        flip_q.add_qtype(qtype)
        return flip_q
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""

        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        conj_q = QHa(qtype=self.qtype)

        if conj_type == '-':
            conj_q.a[0] = 1.0 * t
            conj_q.a[1] = -1.0 * x
            conj_q.a[2] = -1.0 * y
            conj_q.a[3] = -1.0 * z
            qtype += "*-"

        if conj_type == "'":
            conj_q.a[0] = 1.0 * t
            conj_q.a[1] = -1.0 * x
            conj_q.a[2] = -1.0 * y
            conj_q.a[3] = 1.0 * z
            qtype += "*'"
            
        if conj_type == '*':
            conj_q.a[0] = 1.0 * t
            conj_q.a[1] = 1.0 * x
            conj_q.a[2] = 1.0 * y
            conj_q.a[3] = -1.0 * z
            qtype += "*"
            
        conj_q.add_qtype(qtype)
        return conj_q

    def add_qtype(self, qtype):
        """Adds a qtype to an exiting qtype."""
        
        self.qtype += "." + qtype
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        s_t, s_x, s_y, s_z = self.a[0], self.a[1], self.a[2], self.a[3]
        q1_t, q1_x, q1_y, q1_z = q1.a[0], q1.a[1], q1.a[2], q1.a[3]

        products = {'tt': s_t * q1_t,
                    'xx+yy+zz': s_x * q1_x + s_y * q1_y + s_z * q1_z,
                    'tx+xt': s_t * q1_x + s_x * q1_t,
                    'ty+yt': s_t * q1_y + s_y * q1_t,
                    'tz+zt': s_t * q1_z + s_z * q1_t}

        return products

    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        s_x, s_y, s_z = self.a[1], self.a[2], self.a[3]
        q1_x, q1_y, q1_z = q1.a[1], q1.a[2], q1.a[3]

        products = {'yz-zy': s_y * q1_z - s_z * q1_y,
                    'zx-xz': s_z * q1_x - s_x * q1_z,
                    'xy-yx': s_x * q1_y - s_y * q1_x
                    }

        return products

    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))

        return products

    def square(self, qtype="sq"):
        """Square a quaternion."""

        qxq = self._commuting_products(self)

        sq_q = QHa(qtype=self.qtype)
        sq_q.a[0] = qxq['tt'] - qxq['xx+yy+zz']
        sq_q.a[1] = qxq['tx+xt']
        sq_q.a[2] = qxq['ty+yt']
        sq_q.a[3] = qxq['tz+zt']

        sq_q.add_qtype(qtype)
        return sq_q

    def norm_squared(self, qtype="norm_squared"):
        """The norm_squared of a quaternion."""

        qxq = self._commuting_products(self)

        n_q = QHa(qtype=self.qtype)
        n_q.a[0] = qxq['tt'] + qxq['xx+yy+zz']

        n_q.add_qtype(qtype)
        return n_q

    def norm_squared_of_vector(self, qtype="norm_squaredV"):
        """The norm_squared of the vector of a quaternion."""

        qxq = self._commuting_products(self)

        nv_q = QHa(qtype=self.qtype)
        nv_q.a[0] = qxq['xx+yy+zz']

        nv_q.add_qtype(qtype)
        return nv_q

    def abs_of_q(self, qtype="abs"):
        """The absolute value, the square root of the norm_squared."""

        ns = self.norm_squared()
        sqrt_t = ns.a[0] ** 0.5
        ns.a[0] = sqrt_t

        ns.qtype = self.qtype
        ns.add_qtype(qtype)
        return ns

    def abs_of_vector(self, qtype="absV"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        av = self.norm_squared_of_vector()
        sqrt_t = av.a[0] ** 0.5
        av.a[0] = sqrt_t

        av.qtype = self.qtype
        av.add_qtype(qtype)
        return av

    def add(self, QHa_1, qtype=""):
        """Form a add given 2 quaternions."""

        t_1, x_1, y_1, z_1 = self.a[0], self.a[1], self.a[2], self.a[3]
        t_2, x_2, y_2, z_2 = QHa_1.a[0], QHa_1.a[1], QHa_1.a[2], QHa_1.a[3]

        add_q = QHa()
        add_q.a[0] = t_1 + t_2
        add_q.a[1] = x_1 + x_2
        add_q.a[2] = y_1 + y_2
        add_q.a[3] = z_1 + z_2
        
        if qtype:
            add_q.qtype = qtype
        else:
            add_q.qtype = "{f}+{s}".format(f=self.qtype, s=QHa_1.qtype)
        
        return add_q    

    def dif(self, QHa_1, qtype=""):
        """Form a add given 2 quaternions."""

        t_1, x_1, y_1, z_1 = self.a[0], self.a[1], self.a[2], self.a[3]
        t_2, x_2, y_2, z_2 = QHa_1.a[0], QHa_1.a[1], QHa_1.a[2], QHa_1.a[3]

        dif_q = QHa()
        dif_q.a[0] = t_1 - t_2
        dif_q.a[1] = x_1 - x_2
        dif_q.a[2] = y_1 - y_2
        dif_q.a[3] = z_1 - z_2

        if qtype:
            dif_q.qtype = qtype
        else:
            dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=QHa_1.qtype)
            
        return dif_q

    def product(self, q1, kind="", qtype=""):
        """Form a product given 2 quaternions: standard, even, odd, and even_minus_odd."""

        commuting = self._commuting_products(q1)
        q_even = QHa()
        q_even.a[0] = commuting['tt'] - commuting['xx+yy+zz']
        q_even.a[1] = commuting['tx+xt']
        q_even.a[2] = commuting['ty+yt']
        q_even.a[3] = commuting['tz+zt']
        qxq = self._all_products(q1)
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = QHa()
        q_odd.a[0] = 0
        q_odd.a[1] = anti_commuting['yz-zy']
        q_odd.a[2] = anti_commuting['zx-xz']
        q_odd.a[3] = anti_commuting['xy-yx']
        
        result = QHa()
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'.")
            
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
            
        return result
    
    def Euclidean_product(self, q1, kind="", qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        pq = QHa()
        pq = self.conj().product(q1, kind)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """The inverse of a quaternion."""

        q_conj = self.conj()
        q_norm_squared = self.norm_squared()

        if q_norm_squared.a[0] == 0:
            print("oops, zero on the norm_squared.")
            return self.q0()

        q_norm_squared_inv = QHa([1.0 / q_norm_squared.a[0], 0, 0, 0])
        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        
        q_inv.add_qtype(qtype)
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""
        
        q1_inv = q1.invert()
        q_div = self.product(q1.invert())
        
        if qtype:
            q_div.qtype = qtype
        else:
            q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
            
        return q_div

    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""

        triple = self.product(q1).product(q2)
        return triple

    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1=0, a_2=0, a_3=0, qtype="rot"):
        """Do a rotation given up to three angles."""

        u = QHa([0, a_1, a_2, a_3])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
  
        q_rot.qtype = self.qtype
        q_rot.add_qtype(qtype)
        return q_rot

    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a))
    def boost(self, beta_x=0.0, beta_y=0.0, beta_z=0.0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""

        boost = QHa(sr_gamma_betas(beta_x, beta_y, beta_z))      
        b_conj = boost.conj()

        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
      
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(QHa([0.5, 0.0, 0.0, 0.0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        
        triple_123.add_qtype(qtype)
        return triple_123

    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    # g_form is the form of the function, either minimal or exponential
    # Minimal is what is needed to pass all weak field tests of gravity
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""

        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self

        g_q = QHa(qtype=self.qtype)
        g_q.a[0] = self.a[0] / g_factor
        g_q.a[1] = self.a[1] * g_factor
        g_q.a[2] = self.a[2] * g_factor
        g_q.a[3] = self.a[3] * g_factor

        g_q.add_qtype(qtype)
        return g_q
    
    def q_sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""

        q_out = QHa(qtype=self.qtype)
        v_norm = self.norm_squared_of_vector()
        
        if v_norm.a[0] == 0:
            
            q_out.a[0] = sp.sin(self.a[0])
            q_out.a[1] = 0.0
            q_out.a[2] = 0.0
            q_out.a[3] = 0.0
            
        else:
            
            v_factor = sp.cos(self.a[0]) * sp.sinh(v_norm.a[0]) / v_norm.a[0]
            
            q_out.a[0] = math.trunc(sp.sin(self.a[0]) * sp.cosh(v_norm.t))
            q_out.a[1] = math.trunc(v_factor * self.a[1])
            q_out.a[2] = math.trunc(v_factor * self.a[2])
            q_out.a[3] = math.trunc(v_factor * self.a[3])

        q_out.add_qtype(qtype)
        return q_out


# In[7]:


class TestQHa(unittest.TestCase):
    """Class to make sure all the functions work as expected."""

    Q = QHa([1.0, -2.0, -3.0, -4.0], qtype="Q")
    P = QHa([0.0, 4.0, -3.0, 0.0], qtype="P")
    verbose = True

    def test_qt(self):
        q1 = self.Q.dupe()
        self.assertTrue(q1.a[0] == 1)

    def test_q_0(self):
        q1 = self.Q.dupe()
        q_z = q1.q_0()
        if self.verbose: print("q_0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_1(self):
        q1 = self.Q.dupe()
        q_z = q1.q_1()
        if self.verbose: print("q_1: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
    
    def test_q_i(self):
        q1 = self.Q.dupe()
        q_z = q1.q_i()
        if self.verbose: print("q_i: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_j(self):
        q1 = self.Q.dupe()
        q_z = q1.q_j()
        if self.verbose: print("q_1: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 1)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_k(self):
        q1 = self.Q.dupe()
        q_z = q1.q_k()
        if self.verbose: print("q_k: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 1)
        
    def test_equals(self):
        self.assertTrue(self.Q.equals(self.Q))
        self.assertFalse(self.Q.equals(self.P))

    def test_conj_0(self):
        q1 = self.Q.dupe()
        q_z = q1.conj()
        if self.verbose: print("q_conj 0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_conj_1(self):
        q1 = self.Q.dupe()
        q_z = q1.conj(1)
        if self.verbose: print("q_conj 1: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_conj_2(self):
        q1 = self.Q.dupe()
        q_z = q1.conj(2)
        if self.verbose: print("q_conj 2: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_minus(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj()
        if self.verbose: print("q_vahlen_conj -: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_star(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj('*')
        if self.verbose: print("q_vahlen_conj *: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_prime(self):
        q1 = self.Q.dupe()
        q_z = q1.vahlen_conj("'")
        if self.verbose: print("q_vahlen_conj ': {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == -4)

    def test_square(self):
        q1 = self.Q.dupe()
        q_z = q1.square()
        if self.verbose: print("square: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -28)
        self.assertTrue(q_z.a[1] == -4)
        self.assertTrue(q_z.a[2] == -6)
        self.assertTrue(q_z.a[3] == -8)

    def test_norm_squared(self):
        q1 = self.Q.dupe()
        q_z = q1.norm_squared()
        if self.verbose: print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 30)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_norm_squared_of_vector(self):
        q1 = self.Q.dupe()
        q_z = q1.norm_squared_of_vector()
        if self.verbose: print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 29)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        
    def test_abs_of_q(self):
        q2 = self.P.dupe()
        q_z = q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5.0)
        self.assertTrue(q_z.a[1] == 0.0)
        self.assertTrue(q_z.a[2] == 0.0)
        self.assertTrue(q_z.a[3] == 0.0)
        
    def test_abs_of_vector(self):
        q2 = self.P.dupe()
        q_z = q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        
    def test_add(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.add(q2)
        if self.verbose: print("add: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == -6)
        self.assertTrue(q_z.a[3] == -4)
        
    def test_dif(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.dif(q2)
        if self.verbose: print("dif: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -6)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == -4) 

    def test_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2)
        if self.verbose: print("product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == -8)
        self.assertTrue(q_z.a[2] == -19)
        self.assertTrue(q_z.a[3] == 18)
        
    def test_product_even(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="even")
        if self.verbose: print("product even: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 4)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 0)
        
    def test_product_odd(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="odd")
        if self.verbose: print("product odd: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == -12)
        self.assertTrue(q_z.a[2] == -16)
        self.assertTrue(q_z.a[3] == 18)

    def test_product_even_minus_odd(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.product(q2, kind="even_minus_odd")
        if self.verbose: print("product even_minus_odd: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 16)
        self.assertTrue(q_z.a[2] == 13)
        self.assertTrue(q_z.a[3] == -18)
        
    def test_Euclidean_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.Euclidean_product(q2)
        if self.verbose: print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 16)
        self.assertTrue(q_z.a[2] == 13)
        self.assertTrue(q_z.a[3] == -18)

    def test_invert(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q2.invert()
        if self.verbose: print("invert: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == -0.16)
        self.assertTrue(q_z.a[2] == 0.12)
        self.assertTrue(q_z.a[3] == 0)
                
    def test_divide_by(self):
        q1 = self.Q.dupe()
        q_z = q1.divide_by(q1)
        if self.verbose: print("divide_by: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0) 
        
    def test_triple_product(self):
        q1 = self.Q.dupe()
        q2 = self.P.dupe()
        q_z = q1.triple_product(q2, q1)
        if self.verbose: print("triple product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == -2)
        self.assertTrue(q_z.a[1] == 124)
        self.assertTrue(q_z.a[2] == -84)
        self.assertTrue(q_z.a[3] == 8)
        
    def test_rotate(self):
        q1 = self.Q.dupe()
        q_z = q1.rotate(1)
        if self.verbose: print("rotate: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)
        
    def test_boost(self):
        q1 = self.Q.dupe()
        q1_sq = q1.square()
        q_z = q1.boost(0.003)
        q_z2 = q_z.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(q_z))
        if self.verbose: print("boosted squared: {}".format(q_z2))
        print("{}")
        self.assertTrue(round(q_z2.a[0], 5) == round(q1_sq.a[0], 5))

    def test_g_shift(self):
        q1 = self.Q.dupe()
        q1_sq = q1.square()
        q_z = q1.g_shift(0.003)
        q_z2 = q_z.square()
        q_z_minimal = q1.g_shift(0.003, g_form="minimal")
        q_z2_minimal = q_z_minimal.square()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(q_z))
        if self.verbose: print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.a[0] != q1_sq.a[0])
        self.assertTrue(q_z2.a[1] == q1_sq.a[1])
        self.assertTrue(q_z2.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2.a[3] == q1_sq.a[3])
        self.assertTrue(q_z2_minimal.a[0] != q1_sq.a[0])
        self.assertTrue(q_z2_minimal.a[1] == q1_sq.a[1])
        self.assertTrue(q_z2_minimal.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2_minimal.a[3] == q1_sq.a[3])
        
    def test_q_sin(self):
        QHa([0, 0, 0, 0]).q_sin()


# In[8]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHa())
unittest.TextTestRunner().run(suite);


# ## Using More Numbers via Doublets

# My long term goal is to deal with quaternions on a quaternion manifold. This will have 4 pairs of doublets. Each doublet is paired with its additive inverse. Instead of using real numbers, one uses (3, 0) and (0, 2) to represent +3 and -2 respectively. Numbers such as (5, 6) are allowed. That can be "reduced" to (0, 1).  My sense is that somewhere deep in the depths of relativistic quantum field theory, this will be a "good thing". For now, it is a minor pain to program.

# In[9]:


class Doublet(object):
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
        
        elif isinstance(numbers, sp.Symbol):
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
                    self.p = sp.Symbol(n_list[0])
                    self.n = 0
                      
            if (len(n_list) == 2):
                if n_list[0].isnumeric():
                    self.p = float(n_list[0])
                else:
                    self.p = sp.Symbol(n_list[0])
                    
                if n_list[1].isnumeric():
                    self.n = float(n_list[1])
                else:
                    self.n = sp.Symbol(n_list[1])
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
        
    def d_additive_inverse_up_to_an_automorphism(self, n=0):
        """Creates one additive inverses up to an arbitrary positive n."""
        
        if n == 0:
            return Doublet([self.n + n, self.p + n])
        else:
            red = self.d_reduce()
            return Doublet([red.n + n, red.p +n])
                        
    def d_dif(self, d1, n=0):
        """Take the difference by flipping and adding."""
        d2 = d1.d_additive_inverse_up_to_an_automorphism(n)
                        
        return self.d_add(d2)
        
    def Z2_product(self, d1):
        """Uset the Abelian cyclic group Z2 to form the product of 2 doublets."""
        p1 = self.p * d1.p + self.n * d1.n
        n1 = self.p * d1.n + self.n * d1.p
        
        return Doublet([p1, n1])


# In[10]:


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
        
    def test_d_additive_inverse_up_to_an_automorphism(self):
        d_f = self.d2.d_additive_inverse_up_to_an_automorphism()
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


# In[11]:


suite = unittest.TestLoader().loadTestsFromModule(TestDoublet())
unittest.TextTestRunner().run(suite);


# Repeat the exercise for arrays.

# In[12]:


class Doubleta(object):
    """A pair of number that are additive inverses. It can take
    ints, floats, Symbols, or strings."""
    
    def __init__(self, numbers=None):
        
        if numbers is None:
            self.d = np.array([0.0, 0.0])
            
        elif isinstance(numbers, (int, float)):
            if numbers < 0:
                self.d = np.array([0, -1 * numbers])
            else:
                self.d = np.array([numbers, 0])
                        
        elif isinstance(numbers, sp.Symbol):
            self.d = np.array([numbers, 0])
            
        elif isinstance(numbers, list):
            
            if len(numbers) == 2:
                self.d = np.array([numbers[0], numbers[1]])
                      
        elif isinstance(numbers, str):
            n_list = numbers.split()
            
            if (len(n_list) == 1):
                if n_list.isnumeric():
                    n_value = float(numbers)
                      
                    if n_value < 0:
                        self.d = np.array([0, -1 * n_list[0]])
                      
                    else:
                        self.d = np.array([n_list[0], 0])
                        
                else:
                    self.d = np.array([sp.Symbol(n_list[0]), 0])
                      
            if (len(n_list) == 2):
                if n_list[0].isnumeric():
                    self.d = np.array([float(n_list[0]), float(n_list[1])])
                else:
                    self.d = np.array([sp.Symbol(n_list[0]), sp.Symbol(n_list[1])]) 
        else:
            print ("unable to parse this Double.")

    def __str__(self):
        """Customize the output."""
        return "{p}p  {n}n".format(p=self.d[0], n=self.d[1])
        
    def d_add(self, d1):
        """Add a doublet to another."""
                        
        pa0, n0 = self.d[0], self.d[1]
        p1, n1 = d1.d[0], d1.d[1]
                        
        return Doubleta([pa0 + p1, n0 + n1])

    def d_reduce(self):
        """If p and n are not zero, subtract """
        if self.d[0] == 0 or self.d[1] == 0:
            return Doubleta([self.d[0], self.d[1]])
        
        elif self.d[0] > self.d[1]:
            return Doubleta([self.d[0] - self.d[1], 0])
        
        elif self.d[0] < self.d[1]:
            return Doubleta([0, self.d[1] - self.d[0]])
        
        else:
            return Doubleta()
        
    def d_additive_inverse_up_to_an_automorphism(self, n=0):
        """Creates one additive inverses up to an arbitrary positive n."""
        
        if n == 0:
            return Doubleta([self.d[1], self.d[0]])
        else:
            red = self.d_reduce()
            return Doubleta([red.d[1] + n, red.d[0] + n])
                        
    def d_dif(self, d1, n=0):
        """Take the difference by flipping and adding."""
        d2 = d1.d_additive_inverse_up_to_an_automorphism(n)
                        
        return self.d_add(d2)
        
    def Z2_product(self, d1):
        """Uset the Abelian cyclic group Z2 to form the product of 2 doublets."""
        p1 = self.d[0] * d1.d[0] + self.d[1] * d1.d[1]
        n1 = self.d[0] * d1.d[1] + self.d[1] * d1.d[0]
        
        return Doubleta([p1, n1])


# In[13]:


class TestDoubleta(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    d1 = Doubleta()
    d2 = Doubleta(2)
    d3 = Doubleta(-3)
    dstr12 = Doubleta("1 2")
    dstr13 = Doubleta("3 2")
    
    def test_null(self):
        self.assertTrue(self.d1.d[0] == 0)
        self.assertTrue(self.d1.d[1] == 0)
       
    def test_2(self):
        self.assertTrue(self.d2.d[0] == 2)
        self.assertTrue(self.d2.d[1] == 0)
        
    def test_3(self):
        self.assertTrue(self.d3.d[0] == 0)
        self.assertTrue(self.d3.d[1] == 3)
    
    def test_str12(self):
        self.assertTrue(self.dstr12.d[0] == 1)
        self.assertTrue(self.dstr12.d[1] == 2)
    
    def test_add(self):
        d_add = self.d2.d_add(self.d3)
        self.assertTrue(d_add.d[0] == 2)
        self.assertTrue(d_add.d[1] == 3)
        
    def test_d_additive_inverse_up_to_an_automorphism(self):
        d_f = self.d2.d_additive_inverse_up_to_an_automorphism()
        self.assertTrue(d_f.d[0] == 0)
        self.assertTrue(d_f.d[1] == 2)
        
    def test_dif(self):
        d_d = self.d2.d_dif(self.d3)
        self.assertTrue(d_d.d[0] == 5)
        self.assertTrue(d_d.d[1] == 0)
            
    def test_reduce(self):
        d_add = self.d2.d_add(self.d3)
        d_r = d_add.d_reduce()
        self.assertTrue(d_r.d[0] == 0)
        self.assertTrue(d_r.d[1] == 1)
        
    def test_Z2_product(self):
        Z2p = self.dstr12.Z2_product(self.dstr13)
        self.assertTrue(Z2p.d[0] == 7)
        self.assertTrue(Z2p.d[1] == 8)
        
    def test_reduced_product(self):
        """Reduce before or after, should make no difference."""
        Z2p_1 = self.dstr12.Z2_product(self.dstr13)
        Z2p_red = Z2p_1.d_reduce()
        d_r_1 = self.dstr12.d_reduce()
        d_r_2 = self.dstr13.d_reduce()
        Z2p_2 = d_r_1.Z2_product(d_r_2)
        self.assertTrue(Z2p_red.d[0] == Z2p_2.d[0])
        self.assertTrue(Z2p_red.d[1] == Z2p_2.d[1])


# In[14]:


suite = unittest.TestLoader().loadTestsFromModule(TestDoubleta())
unittest.TextTestRunner().run(suite);


# ## Quaternion Group Q8

# Write a class to handle quaternions given 8 numbers.

# In[15]:


class Q8(object):
    """Quaternions on a quaternion manifold or space-time numbers."""

    def __init__(self, values=None, qtype="Q"):
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
                
        self.qtype=qtype
                
    def __str__(self):
        """Customize the output."""
        return "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(tp=self.dt.p, tn=self.dt.n, 
                                                             xp=self.dx.p, xn=self.dx.n, 
                                                             yp=self.dy.p, yn=self.dy.n, 
                                                             zp=self.dz.p, zn=self.dz.n,
                                                             qt=self.qtype)
    def q4(self):
        """Return a 4 element array."""
        return [self.dt.p - self.dt.n, self.dx.p - self.dx.n, self.dy.p - self.dy.n, self.dz.p - self.dz.n]
        
    def add_qtype(self, qtype):
            """Adds a qtype to an existing one."""
            
            self.qtype += "." + qtype
            
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""
        
        return Q8()
      
    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""
        
        return Q8([1, 0, 0, 0])
    
    def q_i(self, qtype="i"):
        """Return i."""
        
        return Q8([0, 1, 0, 0])
    
    def q_j(self, qtype="j"):
        """Return j."""
        
        return Q8([0, 0, 1, 0])
    
    def q_k(self, qtype="k"):
        """Return k."""
        
        return Q8([0, 0, 0, 1])
    
    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        def equal_doublets(d1, d2):
            d1_red = d1.d_reduce()
            d2_red = d2.d_reduce()
            
            if (d1_red.p == d2_red.p) and (d1_red.n == d2_red.n):
                return True
            
            else:
                return False
            
        if equal_doublets(self.dt, q2.dt) and equal_doublets(self.dx, q2.dx) and equal_doublets(self.dy, q2.dy) and equal_doublets(self.dz, q2.dz):
            return True
        
        else:
            return False
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""
        
        conjq = Q8(qtype=self.qtype)

        if conj_type == 0:
            conjq.dt = self.dt
            conjq.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conjq.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conjq.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
        
        if conj_type == 1:
            conjq.dt = self.dt.d_additive_inverse_up_to_an_automorphism()
            conjq.dx = self.dx
            conjq.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conjq.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            qtype += "1"
            
        if conj_type == 2:
            conjq.dt = self.dt.d_additive_inverse_up_to_an_automorphism()
            conjq.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conjq.dy = self.dy
            conjq.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            qtype += "2"
            
        conjq.add_qtype(qtype)
        return conjq
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""
        conjq = Q8(qtype=self.qtype)

        if conj_type == "-":
            conjq.dt = self.dt
            conjq.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conjq.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conjq.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            qtype += "-"
            
        if conj_type == "'":
            conjq.dt = self.dt
            conjq.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conjq.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conjq.dz = self.dz
            qtype += "'"
            
        if conj_type == "*":
            conjq.dt = self.dt
            conjq.dx = self.dx
            conjq.dy = self.dy
            conjq.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            qtype += "*"
            
        conjq.add_qtype(qtype)
        return conjq

    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt': self.dt.Z2_product(q1.dt),
                    'xx+yy+zz': self.dx.Z2_product(q1.dx).d_add(self.dy.Z2_product(q1.dy)).d_add(self.dz.Z2_product(q1.dz)),
        
                    'tx+xt': self.dt.Z2_product(q1.dx).d_add(self.dx.Z2_product(q1.dt)),
                    'ty+yt': self.dt.Z2_product(q1.dy).d_add(self.dy.Z2_product(q1.dt)),
                    'tz+zt': self.dt.Z2_product(q1.dz).d_add(self.dz.Z2_product(q1.dt))}
        
        return products
    
    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        products = {'yz-zy': self.dy.Z2_product(q1.dz).d_dif(self.dz.Z2_product(q1.dy)),
                    'zx-xz': self.dz.Z2_product(q1.dx).d_dif(self.dx.Z2_product(q1.dz)),
                    'xy-yx': self.dx.Z2_product(q1.dy).d_dif(self.dy.Z2_product(q1.dx))}
        
        return products
    
    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self.anti_commuting_products(q1))
        
        return products
    
    def square(self, qtype=""):
        """Square a quaternion."""
        
        qxq = self._commuting_products(self)
        
        sq_q = Q8(qtype=self.qtype)        
        sq_q.dt = qxq['tt'].d_dif(qxq['xx+yy+zz'])
        sq_q.dx = qxq['tx+xt']
        sq_q.dy = qxq['ty+yt']
        sq_q.dz = qxq['tz+zt']
        
        if qtype:
            sq_q.qtype = qtype
        else:
            sq_q.add_qtype("{s}_sq".format(s=self.qtype))
        return sq_q
    
    def reduce(self, qtype="reduce"):
        """Put all doublets into the reduced form so one of each pair is zero."""

        q_red = Q8(qtype=self.qtype)
        q_red.dt = self.dt.d_reduce()
        q_red.dx = self.dx.d_reduce()
        q_red.dy = self.dy.d_reduce()
        q_red.dz = self.dz.d_reduce()
        
        q_red.add_qtype(qtype)
        return q_red
    
    def norm_squared(self, qtype="norm_squared"):
        """The norm_squared of a quaternion."""
        
        qxq = self._commuting_products(self)
        
        n_q = Q8(qtype=self.qtype)        
        n_q.dt = qxq['tt'].d_add(qxq['xx+yy+zz'])

        n_q.add_qtype(qtype)
        return n_q
    
    def norm_squared_of_vector(self, qtype="norm_squaredV"):
        """The norm_squared of the vector of a quaternion."""
        
        qxq = self._commuting_products(self)
        
        nv_q = Q8(qtype=self.qtype)
        nv_q.dt = qxq['xx+yy+zz']

        nv_q.add_qtype(qtype)
        return nv_q
    
    def abs_of_q(self, qtype="abs"):
        """The absolute value, the square root of the norm_squared."""

        a = self.norm_squared(qtype=self.qtype)
        sqrt_t = a.dt.p ** (1/2)
        a.dt = Doublet(sqrt_t)
        
        a.add_qtype(qtype)
        return a

    def abs_of_vector(self, qtype="absV"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        av = self.norm_squared_of_vector()
        sqrt_t = av.dt.p ** (1/2)
        av.dt = Doublet(sqrt_t)
        
        av.qtype = self.qtype
        av.add_qtype(qtype)
        return av
    
    def add(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        add_q = Q8()
        add_q.dt = self.dt.d_add(q1.dt)
        add_q.dx = self.dx.d_add(q1.dx)
        add_q.dy = self.dy.d_add(q1.dy)
        add_q.dz = self.dz.d_add(q1.dz)
                    
        if qtype:
            add_q.qtype = qtype
        else:
            add_q.qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
            
        return add_q    

    def dif(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        dif_q = Q8()
        dif_q.dt = self.dt.d_dif(q1.dt)
        dif_q.dx = self.dx.d_dif(q1.dx)
        dif_q.dy = self.dy.d_dif(q1.dy)
        dif_q.dz = self.dz.d_dif(q1.dz)
                    
        if qtype:
            dif_q.qtype = qtype
        else:
            dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
            
        return dif_q
    
    def product(self, q1, kind="", qtype=""):
        """Form a product given 2 quaternions: standard, even, odd, and even_minus_odd."""
    
        commuting = self._commuting_products(q1)
        q_even = Q8()
        q_even.dt = commuting['tt'].d_dif(commuting['xx+yy+zz'])
        q_even.dx = commuting['tx+xt']
        q_even.dy = commuting['ty+yt']
        q_even.dz = commuting['tz+zt']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = Q8()
        q_odd.dt = Doublet()
        q_odd.dx = anti_commuting['yz-zy']
        q_odd.dy = anti_commuting['zx-xz']
        q_odd.dz = anti_commuting['xy-yx']
        
        result = Q8()
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Fouf 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'")
            
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
            
        return result
    
    def Euclidean_product(self, q1, kind="", qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        pq = Q8()
        pq = self.conj().product(q1, kind)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """Invert a quaternion."""
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared().reduce()
        
        if q_norm_squared.dt.p == 0:
            return self.q0()
        
        q_norm_squared_inv = Q8([1.0 / q_norm_squared.dt.p, 0, 0, 0, 0, 0, 0, 0])

        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        
        q_inv.add_qtype(qtype)
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""

        q_inv = q1.invert()
        q_div = self.product(q_inv) 
        
        if qtype:
            q_div.qtype = qtype
        else:
            q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
            
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        triple = self.product(q1).product(q2)
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1p=0, a_1n=0, a_2p=0, a_2n=0, a_3p=0, a_3n=0):
        """Do a rotation given up to three angles."""
    
        u = Q8([0, 0, a_1p, a_1n, a_2p, a_2n, a_3p, a_3n])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""
        
        boost = Q8(sr_gamma_betas(beta_x, beta_y, beta_z))
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Q8([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        
        triple_123.add_qtype(qtype)
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
            if qtype == "g_shift":
                qtype = "g_exp"
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
            if qtype == "g_shift":
                qtype = "g_minimal"
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self
        exp_g = sp.exp(dimensionless_g)
        
        g_q = Q8(qtype=self.qtype)
        g_q.dt = Doublet([self.dt.p / exp_g, self.dt.n / exp_g])
        g_q.dx = Doublet([self.dx.p * exp_g, self.dx.n * exp_g])
        g_q.dy = Doublet([self.dy.p * exp_g, self.dy.n * exp_g])
        g_q.dz = Doublet([self.dz.p * exp_g, self.dz.n * exp_g])
        
        g_q.add_qtype(qtype)
        return g_q


# In[16]:


class TestQ8(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Q8([1, 0, 0, 2, 0, 3, 0, 4])
    q2 = Q8([0, 0, 4, 0, 0, 3, 0, 0])
    q_big = Q8([1, 2, 3, 4, 5, 6, 7, 8])
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1.dt.p == 1)
    
    def test_q_0(self):
        q_z = self.q1.q_0()
        if self.verbose: print("q_0: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_1(self):
        q_z = self.q1.q_1()
        if self.verbose: print("q_1: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_i(self):
        q_z = self.q1.q_i()
        if self.verbose: print("q_i: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 1)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_j(self):
        q_z = self.q1.q_j()
        if self.verbose: print("q_j: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 1)
        self.assertTrue(q_z.dz.p == 0)
                
    def test_q_k(self):
        q_z = self.q1.q_k()
        if self.verbose: print("q_k: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 1)

    def test_equals(self):
        self.assertTrue(self.q1.equals(self.q1))
        self.assertFalse(self.q1.equals(self.q2))
        
    def test_conj_0(self):
        q_z = self.q1.conj()
        if self.verbose: print("conj 0: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_conj_1(self):
        q_z = self.q1.conj(1)
        if self.verbose: print("conj 1: {}".format(q_z))
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_conj_2(self):
        q_z = self.q1.conj(2)
        if self.verbose: print("conj 2: {}".format(q_z))
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 4)
        
    def test_vahlen_conj_0(self):
        q_z = self.q1.vahlen_conj()
        if self.verbose: print("vahlen conj -: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_vahlen_conj_1(self):
        q_z = self.q1.vahlen_conj("'")
        if self.verbose: print("vahlen conj ': {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.n == 4)
                 
    def test_vahlen_conj_2(self):
        q_z = self.q1.vahlen_conj('*')
        if self.verbose: print("vahlen conj *: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 4)
        
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
        
    def test_norm_squared(self):
        q_z = self.q1.norm_squared()
        if self.verbose: print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 30)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_norm_squared_of_vector(self):
        q_z = self.q1.norm_squared_of_vector()
        if self.verbose: print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 29)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_abs_of_q(self):
        q_z = self.q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 5)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_abs_of_vector(self):
        q_z = self.q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 5)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_add(self):
        q_z = self.q1.add(self.q2)
        if self.verbose: print("add: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 4)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 6)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 4)
        
    def test_add_reduce(self):
        q_z_red = self.q1.add(self.q2).reduce()
        if self.verbose: print("add reduce: {}".format(q_z_red))
        self.assertTrue(q_z_red.dt.p == 1)
        self.assertTrue(q_z_red.dt.n == 0)
        self.assertTrue(q_z_red.dx.p == 2)
        self.assertTrue(q_z_red.dx.n == 0)
        self.assertTrue(q_z_red.dy.p == 0)
        self.assertTrue(q_z_red.dy.n == 6)
        self.assertTrue(q_z_red.dz.p == 0)
        self.assertTrue(q_z_red.dz.n == 4)
        
    def test_dif(self):
        q_z = self.q1.dif(self.q2)
        if self.verbose: print("dif: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 6) 
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 4) 

    def test_product(self):
        q_z = self.q1.product(self.q2).reduce()
        if self.verbose: print("product: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 8)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 19)
        self.assertTrue(q_z.dz.p == 18)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_product_even(self):
        q_z = self.q1.product(self.q2, kind="even").reduce()
        if self.verbose: print("product, kind even: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 4)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_product_odd(self):
        q_z = self.q1.product(self.q2, kind="odd").reduce()
        if self.verbose: print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 12)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 16)
        self.assertTrue(q_z.dz.p == 18)
        self.assertTrue(q_z.dz.n == 0)
    
    def test_product_even_minus_odd(self):
        q_z = self.q1.product(self.q2, kind="even_minus_odd").reduce()
        if self.verbose: print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 16)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 13)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 18)
        
    def test_Euclidean_product(self):
        q_z = self.q1.Euclidean_product(self.q2).reduce()
        if self.verbose: print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 16)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 13)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 18)
        
    def test_invert(self):
        q_z = self.q2.invert().reduce()
        if self.verbose: print("inverse: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0.16)
        self.assertTrue(q_z.dy.p == 0.12)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)

    def test_divide_by(self):
        q_z = self.q1.divide_by(self.q1).reduce()
        if self.verbose: print("inverse: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0) 
        
    def test_triple_product(self):
        q_z = self.q1.triple_product(self.q2, self.q1).reduce()
        if self.verbose: print("triple: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 2)
        self.assertTrue(q_z.dx.p == 124)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 84)
        self.assertTrue(q_z.dz.p == 8)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_rotate(self):
        q_z = self.q1.rotate(1).reduce()
        if self.verbose: print("rotate: {}".format(q_z))
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 4)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_boost(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.boost(0.003)
        q_z2 = q_z.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(q_z))
        if self.verbose: print("b squared: {}".format(q_z2))
        self.assertTrue(round(q_z2.dt.n, 12) == round(q1_sq.dt.n, 12))
        
    def test_g_shift(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.g_shift(0.003)
        q_z2 = q_z.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(q_z))
        if self.verbose: print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.dt.n != q1_sq.dt.n)
        self.assertTrue(q_z2.dx.p == q1_sq.dx.p)
        self.assertTrue(q_z2.dx.n == q1_sq.dx.n)
        self.assertTrue(q_z2.dy.p == q1_sq.dy.p)
        self.assertTrue(q_z2.dy.n == q1_sq.dy.n)
        self.assertTrue(q_z2.dz.p == q1_sq.dz.p)
        self.assertTrue(q_z2.dz.n == q1_sq.dz.n)


# In[17]:


suite = unittest.TestLoader().loadTestsFromModule(TestQ8())
unittest.TextTestRunner().run(suite);


# ## Class Q8a as nparrays

# In[18]:


class Q8a(object):
    """Quaternions on a quaternion manifold or space-time numbers."""

    def __init__(self, values=None, qtype="Q"):
        if values is None:
            d_zero = Doubleta()
            self.a = np.array([d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0]])
     
        elif isinstance(values, list):
            if len(values) == 4:
                self.a = np.array([Doubleta(values[0]).d[0], Doubleta(values[0]).d[1], 
                                   Doubleta(values[1]).d[0], Doubleta(values[1]).d[1], 
                                   Doubleta(values[2]).d[0], Doubleta(values[2]).d[1], 
                                   Doubleta(values[3]).d[0], Doubleta(values[3]).d[1]])
        
            if len(values) == 8:
                self.a = np.array([Doubleta([values[0], values[1]]).d[0], Doubleta([values[0], values[1]]).d[1],
                                   Doubleta([values[2], values[3]]).d[0], Doubleta([values[2], values[3]]).d[1],
                                   Doubleta([values[4], values[5]]).d[0], Doubleta([values[4], values[5]]).d[1],
                                   Doubleta([values[6], values[7]]).d[0], Doubleta([values[6], values[7]]).d[1]])
                                  
        self.qtype=qtype
                
    def __str__(self):
        """Customize the output."""
        return "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(tp=self.a[0], tn=self.a[1], 
                                                             xp=self.a[2], xn=self.a[3], 
                                                             yp=self.a[4], yn=self.a[5], 
                                                             zp=self.a[6], zn=self.a[7],
                                                             qt=self.qtype)
    def q4(self):
        """Return a 4 element array."""
        return [self.a[0] - self.a[1], self.a[0] - self.a[1], self.a[4] - self.a[5], self.a[6] - self.a[7]]
        
    def add_qtype(self, qtype):
            """Adds a qtype to an existing one."""
            
            self.qtype += "." + qtype
            
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""
        
        return Q8a()
      
    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""
        
        return Q8a([1, 0, 0, 0, 0, 0, 0, 0])
    
    def q_i(self, qtype="i"):
        """Return i."""
        
        return Q8a([0, 0, 1, 0, 0, 0, 0, 0])
    
    def q_j(self, qtype="One"):
        """Return j."""
        
        return Q8a([0, 0, 0, 0, 1, 0, 0, 0])
    

    def q_k(self, qtype="k"):
        """Return k."""
        
        return Q8a([0, 0, 0, 0, 0, 0, 1, 0])

    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        self_red = self.reduce()
        q2_red = q2.reduce()
        result = True
        
        for i in range(8):
            if self_red.a[i] != q2_red.a[i]:
                result = False
        
        return result
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""
        
        conj_q = Q8a(qtype=self.qtype)

        # Flip all but t.                          
        if conj_type == 0:
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
        
        # Flip all but x.
        if conj_type == 1:
            conj_q.a[0] = self.a[1]
            conj_q.a[1] = self.a[0]
            conj_q.a[2] = self.a[2]
            conj_q.a[3] = self.a[3]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "*1"

        # Flip all but y.                                 
        if conj_type == 2:
            conj_q.a[0] = self.a[1]
            conj_q.a[1] = self.a[0]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[4]
            conj_q.a[5] = self.a[5]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "*2"
            
        conj_q.add_qtype(qtype)
        return conj_q
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""
        conj_q = Q8a(qtype=self.qtype)

        if conj_type == "-":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "-"

        # Flip the sign of x and y.
        if conj_type == "'":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[6]
            conj_q.a[7] = self.a[7]
            qtype += "'"
            
        # Flip the sign of only z.
        if conj_type == "*":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[2]
            conj_q.a[3] = self.a[3]
            conj_q.a[4] = self.a[4]
            conj_q.a[5] = self.a[5]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "*"
            
        conj_q.add_qtype(qtype)
        return conj_q

    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt0': self.a[0] * q1.a[0] + self.a[1] * q1.a[1],
                    'tt1': self.a[0] * q1.a[1] + self.a[1] * q1.a[0],
                    
                    'xx+yy+zz0': self.a[2] * q1.a[2] + self.a[3] * q1.a[3] + self.a[4] * q1.a[4] + self.a[5] * q1.a[5] + self.a[6] * q1.a[6] + self.a[7] * q1.a[7], 
                    'xx+yy+zz1': self.a[2] * q1.a[3] + self.a[3] * q1.a[2] + self.a[4] * q1.a[5] + self.a[5] * q1.a[4] + self.a[6] * q1.a[7] + self.a[7] * q1.a[6], 
                    
                    'tx+xt0': self.a[0] * q1.a[2] + self.a[1] * q1.a[3] + self.a[2] * q1.a[0] + self.a[3] * q1.a[1],
                    'tx+xt1': self.a[0] * q1.a[3] + self.a[1] * q1.a[2] + self.a[3] * q1.a[0] + self.a[2] * q1.a[1],
                    
                    'ty+yt0': self.a[0] * q1.a[4] + self.a[1] * q1.a[5] + self.a[4] * q1.a[0] + self.a[5] * q1.a[1],
                    'ty+yt1': self.a[0] * q1.a[5] + self.a[1] * q1.a[4] + self.a[5] * q1.a[0] + self.a[4] * q1.a[1],
                    
                    'tz+zt0': self.a[0] * q1.a[6] + self.a[1] * q1.a[7] + self.a[6] * q1.a[0] + self.a[7] * q1.a[1],
                    'tz+zt1': self.a[0] * q1.a[7] + self.a[1] * q1.a[6] + self.a[7] * q1.a[0] + self.a[6] * q1.a[1]
                    }
        
        return products
    
    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        yz0 = self.a[4] * q1.a[6] + self.a[5] * q1.a[7]
        yz1 = self.a[4] * q1.a[7] + self.a[5] * q1.a[6]
        zy0 = self.a[6] * q1.a[4] + self.a[7] * q1.a[5]
        zy1 = self.a[6] * q1.a[5] + self.a[7] * q1.a[4]

        zx0 = self.a[6] * q1.a[2] + self.a[7] * q1.a[3]
        zx1 = self.a[6] * q1.a[3] + self.a[7] * q1.a[2]
        xz0 = self.a[2] * q1.a[6] + self.a[3] * q1.a[7]
        xz1 = self.a[2] * q1.a[7] + self.a[3] * q1.a[6]

        xy0 = self.a[2] * q1.a[4] + self.a[3] * q1.a[5]
        xy1 = self.a[2] * q1.a[5] + self.a[3] * q1.a[4]
        yx0 = self.a[4] * q1.a[2] + self.a[5] * q1.a[3]
        yx1 = self.a[4] * q1.a[3] + self.a[5] * q1.a[2]
                                   
        products = {'yz-zy0': yz0 + zy1,
                    'yz-zy1': yz1 + zy0,
                    
                    'zx-xz0': zx0 + xz1,
                    'zx-xz1': zx1 + xz0,
                    
                    'xy-yx0': xy0 + yx1,
                    'xy-yx1': xy1 + yx0}
        
        return products
    
    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))
        
        return products
    
    def square(self, qtype=""):
        """Square a quaternion."""
        
        qxq = self._commuting_products(self)
        
        sq_q = Q8a(qtype=self.qtype)        
        sq_q.a[0] = qxq['tt0'] + (qxq['xx+yy+zz1'])
        sq_q.a[1] = qxq['tt1'] + (qxq['xx+yy+zz0'])
        sq_q.a[2] = qxq['tx+xt0']
        sq_q.a[3] = qxq['tx+xt1']
        sq_q.a[4] = qxq['ty+yt0']
        sq_q.a[5] = qxq['ty+yt1']
        sq_q.a[6] = qxq['tz+zt0']
        sq_q.a[7] = qxq['tz+zt1']
        
        if qtype:
            sq_q.qtype = qtype
        else:
            sq_q.add_qtype("{s}_sq".format(s=self.qtype))
        return sq_q
    
    def reduce(self, qtype="reduce"):
        """Put all Doubletas into the reduced form so one of each pair is zero."""

        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        red_x = Doubleta([self.a[2], self.a[3]]).d_reduce()
        red_y = Doubleta([self.a[4], self.a[5]]).d_reduce()
        red_z = Doubleta([self.a[6], self.a[7]]).d_reduce()
            
        q_red = Q8a(qtype=self.qtype)
        q_red.a[0] = red_t.d[0]
        q_red.a[1] = red_t.d[1]
        q_red.a[2] = red_x.d[0]
        q_red.a[3] = red_x.d[1]
        q_red.a[4] = red_y.d[0]
        q_red.a[5] = red_y.d[1]
        q_red.a[6] = red_z.d[0]
        q_red.a[7] = red_z.d[1]

        q_red.add_qtype(qtype)
        return q_red
    
    def norm_squared(self, qtype="norm_squared"):
        """The norm_squared of a quaternion."""
        
        qxq = self._commuting_products(self)
        
        n_q = Q8a(qtype=self.qtype)        
        n_q.a[0] = qxq['tt0'] + qxq['xx+yy+zz0']

        n_q.add_qtype(qtype)
        return n_q
    
    def norm_squared_of_vector(self, qtype="norm_squaredV"):
        """The norm_squared of the vector of a quaternion."""
        
        qxq = self._commuting_products(self)
        
        nv_q = Q8a(qtype=self.qtype)
        nv_q.a[0] = qxq['xx+yy+zz0']

        nv_q.add_qtype(qtype)
        return nv_q
    
        
    def abs_of_q(self, qtype="abs"):
        """The absolute value, the square root of the norm_squared."""

        abq = self.norm_squared(qtype=self.qtype)
        sqrt_t = abq.a[0] ** (1/2)
        abq.a[0] = sqrt_t
        
        abq.add_qtype(qtype)
        return abq

    def abs_of_vector(self, qtype="absV"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        av = self.norm_squared_of_vector()
        sqrt_t = av.a[0] ** (1/2)
        av.a[0] = sqrt_t
        
        av.qtype = self.qtype
        av.add_qtype(qtype)
        return av
    
    def add(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        add_q = Q8a()
        for i in range(0, 8):
            add_q.a[i] = self.a[i] + q1.a[i]
                    
        if qtype:
            add_q.qtype = qtype
        else:
            add_q.qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
            
        return add_q    

    def dif(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        dif_q = Q8a()

        dif_q.a[0] = self.a[0] + q1.a[1]
        dif_q.a[1] = self.a[1] + q1.a[0]
        dif_q.a[2] = self.a[2] + q1.a[3]
        dif_q.a[3] = self.a[3] + q1.a[2]
        dif_q.a[4] = self.a[4] + q1.a[5]
        dif_q.a[5] = self.a[5] + q1.a[4]
        dif_q.a[6] = self.a[6] + q1.a[7]
        dif_q.a[7] = self.a[7] + q1.a[6]
     
        if qtype:
            dif_q.qtype = qtype
        else:
            dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
            
        return dif_q
    
    def product(self, q1, kind="", qtype=""):
        """Form a product given 2 quaternions."""

        commuting = self._commuting_products(q1)
        q_even = Q8a()
        q_even.a[0] = commuting['tt0'] + commuting['xx+yy+zz1']
        q_even.a[1] = commuting['tt1'] + commuting['xx+yy+zz0']
        q_even.a[2] = commuting['tx+xt0']
        q_even.a[3] = commuting['tx+xt1']
        q_even.a[4] = commuting['ty+yt0']
        q_even.a[5] = commuting['ty+yt1']
        q_even.a[6] = commuting['tz+zt0']
        q_even.a[7] = commuting['tz+zt1']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = Q8a()
        q_odd.a[0] = 0
        q_odd.a[1] = 0
        q_odd.a[2] = anti_commuting['yz-zy0']
        q_odd.a[3] = anti_commuting['yz-zy1']
        q_odd.a[4] = anti_commuting['zx-xz0']
        q_odd.a[5] = anti_commuting['zx-xz1']
        q_odd.a[6] = anti_commuting['xy-yx0']
        q_odd.a[7] = anti_commuting['xy-yx1']

        result = Q8a()
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        else:
            raise Exception("Three 'kind' values are known: '', 'even', and 'odd'")
            
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
            
        return result

    def Euclidean_product(self, q1, kind="", qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        pq = Q8a()
        pq = self.conj().product(q1, kind)

        return pq

    def invert(self, qtype="^-1"):
        """Invert a quaternion."""
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared().reduce()
        
        if q_norm_squared.a[0] == 0:
            return self.q0()
        
        q_norm_squared_inv = Q8a([1.0 / q_norm_squared.a[0], 0, 0, 0, 0, 0, 0, 0])

        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        
        q_inv.add_qtype(qtype)
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""

        q_inv = q1.invert()
        q_div = self.product(q_inv) 
        
        if qtype:
            q_div.qtype = qtype
        else:
            q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
            
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        triple = self.product(q1).product(q2)
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1p=0, a_1n=0, a_2p=0, a_2n=0, a_3p=0, a_3n=0):
        """Do a rotation given up to three angles."""
    
        u = Q8a([0, 0, a_1p, a_1n, a_2p, a_2n, a_3p, a_3n])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""
        
        boost = Q8a(sr_gamma_betas(beta_x, beta_y, beta_z))
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Q8a([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        
        triple_123.add_qtype(qtype)
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
            if qtype == "g_shift":
                qtype = "g_exp"
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
            if qtype == "g_shift":
                qtype = "g_minimal"
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self
        exp_g = sp.exp(dimensionless_g)
        
        dt = Doubleta([self.a[0] / exp_g, self.a[1] / exp_g])
        dx = Doubleta([self.a[2] * exp_g, self.a[3] * exp_g])
        dy = Doubleta([self.a[4] * exp_g, self.a[5] * exp_g])
        dz = Doubleta([self.a[6] * exp_g, self.a[7] * exp_g])
        
        g_q = Q8a(qtype=self.qtype)
        g_q.a[0] = dt.d[0]
        g_q.a[1] = dt.d[1]
        g_q.a[2] = dx.d[0]
        g_q.a[3] = dx.d[1]
        g_q.a[4] = dy.d[0]
        g_q.a[5] = dy.d[1]
        g_q.a[6] = dz.d[0]
        g_q.a[7] = dz.d[1]
        
        g_q.add_qtype(qtype)
        return g_q


# In[19]:


class TestQ8a(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Q8a([1, 0, 0, 2, 0, 3, 0, 4])
    q2 = Q8a([0, 0, 4, 0, 0, 3, 0, 0])
    q_big = Q8a([1, 2, 3, 4, 5, 6, 7, 8])
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1.a[0] == 1)
    
    def test_q_zero(self):
        q_z = self.q1.q_0()
        if self.verbose: print("q0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_1(self):
        q_z = self.q1.q_1()
        if self.verbose: print("q_1: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_i(self):
        q_z = self.q1.q_i()
        if self.verbose: print("q_i: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 1)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_j(self):
        q_z = self.q1.q_j()
        if self.verbose: print("q_j: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 1)
        self.assertTrue(q_z.a[6] == 0)
    
    def test_q_k(self):
        q_z = self.q1.q_k()
        if self.verbose: print("q_k: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 1)
                
    def test_conj_0(self):
        q_z = self.q1.conj()
        if self.verbose: print("conj 0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_equals(self):
        self.assertTrue(self.q1.equals(self.q1))
        self.assertFalse(self.q1.equals(self.q2))
                 
    def test_conj_1(self):
        q_z = self.q1.conj(1)
        if self.verbose: print("conj 1: {}".format(q_z))
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
                 
    def test_conj_2(self):
        q_z = self.q1.conj(2)
        if self.verbose: print("conj 2: {}".format(q_z))
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_vahlen_conj_0(self):
        q_z = self.q1.vahlen_conj()
        if self.verbose: print("vahlen conj -: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
                 
    def test_vahlen_conj_1(self):
        q_z = self.q1.vahlen_conj("'")
        if self.verbose: print("vahlen conj ': {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[7] == 4)
                 
    def test_vahlen_conj_2(self):
        q_z = self.q1.vahlen_conj('*')
        if self.verbose: print("vahlen conj *: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_square(self):
        q_sq = self.q1.square()
        q_sq_red = q_sq.reduce()
        if self.verbose: print("square: {}".format(q_sq))
        if self.verbose: print("square reduced: {}".format(q_sq_red))
        self.assertTrue(q_sq.a[0] == 1)
        self.assertTrue(q_sq.a[1] == 29)
        self.assertTrue(q_sq.a[3] == 4)
        self.assertTrue(q_sq.a[5] == 6)
        self.assertTrue(q_sq.a[7] == 8)
        self.assertTrue(q_sq_red.a[0] == 0)
        self.assertTrue(q_sq_red.a[1] == 28)
                
    def test_reduce(self):
        q_red = self.q_big.reduce()
        if self.verbose: print("q_big reduced: {}".format(q_red))
        self.assertTrue(q_red.a[0] == 0)
        self.assertTrue(q_red.a[1] == 1)
        self.assertTrue(q_red.a[2] == 0)
        self.assertTrue(q_red.a[3] == 1)
        self.assertTrue(q_red.a[4] == 0)
        self.assertTrue(q_red.a[5] == 1)
        self.assertTrue(q_red.a[6] == 0)
        self.assertTrue(q_red.a[7] == 1)
        
    def test_norm_squared(self):
        q_z = self.q1.norm_squared()
        if self.verbose: print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 30)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_norm_squared_of_vector(self):
        q_z = self.q1.norm_squared_of_vector()
        if self.verbose: print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 29)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_abs_of_q(self):
        q_z = self.q2.abs_of_q()
        if self.verbose: print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_abs_of_vector(self):
        q_z = self.q2.abs_of_vector()
        if self.verbose: print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_add(self):
        q_z = self.q1.add(self.q2)
        if self.verbose: print("add: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 4)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 6)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 4)
        
    def test_add_reduce(self):
        q_z_red = self.q1.add(self.q2).reduce()
        if self.verbose: print("add reduce: {}".format(q_z_red))
        self.assertTrue(q_z_red.a[0] == 1)
        self.assertTrue(q_z_red.a[1] == 0)
        self.assertTrue(q_z_red.a[2] == 2)
        self.assertTrue(q_z_red.a[3] == 0)
        self.assertTrue(q_z_red.a[4] == 0)
        self.assertTrue(q_z_red.a[5] == 6)
        self.assertTrue(q_z_red.a[6] == 0)
        self.assertTrue(q_z_red.a[7] == 4)
        
    def test_dif(self):
        q_z = self.q1.dif(self.q2)
        if self.verbose: print("dif: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 6) 
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 4) 

    def test_product(self):
        q_z = self.q1.product(self.q2).reduce()
        if self.verbose: print("product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 8)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 19)
        self.assertTrue(q_z.a[6] == 18)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_product_even(self):
        q_z = self.q1.product(self.q2, kind="even").reduce()
        if self.verbose: print("product, kind even: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 4)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_product_odd(self):
        q_z = self.q1.product(self.q2, kind="odd").reduce()
        if self.verbose: print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 12)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 16)
        self.assertTrue(q_z.a[6] == 18)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_Euclidean_product(self):
        q_z = self.q1.Euclidean_product(self.q2).reduce()
        if self.verbose: print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 16)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 13)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 18)
    
        
    def test_invert(self):
        q_z = self.q2.invert().reduce()
        if self.verbose: print("inverse: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0.16)
        self.assertTrue(q_z.a[4] == 0.12)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)

    def test_divide_by(self):
        q_z = self.q1.divide_by(self.q1).reduce()
        if self.verbose: print("inverse: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0) 
        
    def test_triple_product(self):
        q_z = self.q1.triple_product(self.q2, self.q1).reduce()
        if self.verbose: print("triple: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 124)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 84)
        self.assertTrue(q_z.a[6] == 8)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_rotate(self):
        q_z = self.q1.rotate(1).reduce()
        if self.verbose: print("rotate: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 4)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_boost(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.boost(0.003)
        q_z2 = q_z.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("boosted: {}".format(q_z))
        if self.verbose: print("b squared: {}".format(q_z2))
        self.assertTrue(round(q_z2.a[1], 12) == round(q1_sq.a[1], 12))
        
    def test_g_shift(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.g_shift(0.003)
        q_z2 = q_z.square().reduce()
        if self.verbose: print("q1_sq: {}".format(q1_sq))
        if self.verbose: print("g_shift: {}".format(q_z))
        if self.verbose: print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.a[1] != q1_sq.a[1])
        self.assertTrue(q_z2.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2.a[3] == q1_sq.a[3])
        self.assertTrue(q_z2.a[4] == q1_sq.a[4])
        self.assertTrue(q_z2.a[5] == q1_sq.a[5])
        self.assertTrue(q_z2.a[6] == q1_sq.a[6])
        self.assertTrue(q_z2.a[7] == q1_sq.a[7])


# In[20]:


suite = unittest.TestLoader().loadTestsFromModule(TestQ8a())
unittest.TextTestRunner().run(suite);


# ## Equivalence Classes

# Create a class that can figure out if two quaternions are in the same equivalence class. An equivalence class of space-time is a subset of events in space-time. For example, the future equivalence class would have any event that happens in the future. All time-like events have an interval that is positive.
# 
# A series of images were created to show each class.  For the future, here is the equivalence class:
# ![](images/eq_classes/time_future.png)
# There is a smaller class for those that are exactly the same amount in the future. They have a different icon:
# ![](images/eq_classes/time_future_exact.png)
# Such an exact relation is not of much interest to physicists since Einstein showed that holds for only one set of observers. If one is moving relative to the reference observer, the two events would look like they occured at different times in the future, presuming perfectly accurate measuring devices.
# 

# In[21]:


def round_sig_figs(num, sig_figs):
    """Round to specified number of sigfigs.

    # from http://code.activestate.com/recipes/578114-round-number-to-specified-number-of-significant-di/
    """
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0


# In[22]:


class EQ(object):
    """A class that compairs pairs of quaternions."""
    
    # Read images in once for the class.
    eq_images = {}
    qtd_dir =  os.path.dirname(IPython.utils.path.filefind('Q_tools.ipynb'))
    im_dir = "{qd}/images/eq_classes".format(qd=qtd_dir)
    im_files = "{imd}/*png".format(imd=im_dir)

    for eq_image_file in glob(im_files):
        file_name = basename(eq_image_file)
        eq_class_name = (file_name.split(sep='.'))[0]
        eq_images[eq_class_name] = mpimg.imread(eq_image_file)
        
    def __init__(self, q1, q2, sig_figs=10):
        
        # Convert the quaternions into the Q8 reduced form.
        if isinstance(q1, QH):
            self.q1 = Q8([q1.t, q1.x, q1.y, q1.z])
        
        elif(isinstance(q1, Q8)):
            self.q1 = q1.reduce()
            
        if isinstance(q2, QH):
            self.q2 = Q8([q2.t, q2.x, q2.y, q2.z])
            
        elif(isinstance(q2, Q8)):
            self.q2 = q2.reduce()
                
        # The quaternions used by this class are
        # linear, square, and the norm_squared of a quaternion so do the calculations once.
        
        self.q1_square = self.q1.square().reduce()
        self.q2_square = self.q2.square().reduce()
        
        self.q1_norm_squared_minus_1 = self.q1.norm_squared().dif(self.q1.q_1()).reduce()
        self.q2_norm_squared_minus_1 = self.q2.norm_squared().dif(self.q1.q_1()).reduce()

        # Store results here
        self.classes = {}
        self.sig_figs = sig_figs

    def get_class(self, q1, q2, names, position):
        """A general tool to figure out a scalar class. 
           Names is a dictionary that needs values for 'class', 'positive', 'negative', and 'divider'.
           position needs to be dt, dx, dy or dz"""
        
        q1_d = {'dt': q1.dt, 'dy': q1.dy, 'dx': q1.dx, 'dz': q1.dz}
        q2_d = {'dt': q2.dt, 'dy': q2.dy, 'dx': q2.dx, 'dz': q2.dz}

        
        # Since the quaternions in the Q8 form are reduced just look for non-zero values.
        if q1_d[position].p and q2_d[position].p:

            if round_sig_figs(q1_d[position].p, self.sig_figs) == round_sig_figs(q2_d[position].p, self.sig_figs):
                result = "{np}_exact".format(np=names["positive"])
            else:
                result = "{np}".format(np=names["positive"])
                
        elif q1_d[position].n and q2_d[position].n:
            
            if round_sig_figs(q1_d[position].n, self.sig_figs) == round_sig_figs(q2_d[position].n, self.sig_figs):
                result = "{nn}_exact".format(nn=names["negative"])
            else:
                result = "{nn}".format(nn=names["negative"])
                
        elif not q1_d[position].p and not q1_d[position].n and not q2_d[position].p and not q2_d[position].n:
            result = "{nd}_exact".format(nd=names["divider"])
            
        else:
            result = "disjoint"
            
        self.classes[names["class"]] = result
        return result
    
    def time(self):
        """Figure out time equivalence class."""
        
        names = {'class': 'time', 'positive': 'future', 'negative': 'past', 'divider': 'now'}        
        result = self.get_class(self.q1, self.q2, names, 'dt')
        return result
    
    def space(self):
        """Figure out time equivalence class."""
        
        positions = ['dx', 'dy', 'dz']
        
        names = []
        names.append({'class': 'space-1', 'positive': 'right', 'negative': 'left', 'divider': 'here'})
        names.append({'class': 'space-2', 'positive': 'up', 'negative': 'down', 'divider': 'here'})
        names.append({'class': 'space-3', 'positive': 'near', 'negative': 'far', 'divider': 'here'})
        
        results = []
                     
        for name, position in zip(names, positions):
            results.append(self.get_class(self.q1, self.q2, name, position))
        
        return results
    
    def space_time(self):
        """Do both time and space, return an array."""
        
        results = []
        results.append(self.time())
        results.extend(self.space())
        return results
    
    def causality(self):
        """There is only one causality equivalence class."""
        
        names = {'class': 'causality', 'positive': 'time-like', 'negative': 'space-like', 'divider': 'light-like'}
        result = self.get_class(self.q1_square, self.q2_square, names, 'dt')
        return result
    
    def space_times_time(self):
        """Figure out the space-times-time equivalence class used in the quaternion gravity proposal."""
        
        positions = ['dx', 'dy', 'dz']
        
        names = []
        names.append({'class': 'space-times-time-1', 'positive': 'future-right', 
                      'negative': 'future-left', 'divider': 'here-now'})
        names.append({'class': 'space-times-time-2', 'positive': 'future-up', 
                      'negative': 'future-down', 'divider': 'here-now'})
        names.append({'class': 'space-times-time-3', 'positive': 'future-near', 
                      'negative': 'future-far', 'divider': 'here-now'})
        
        results = []
                     
        for name, position in zip(names, positions):
            results.append(self.get_class(self.q1, self.q2, name, position))
        
        return results

    def squared(self):
        """Return both causality and space_times_time as a list."""

        results = []
        results.append(self.causality())
        results.extend(self.space_times_time())
        return results

    def norm_squared_of_unity(self):
        """Find out if the norm_squared of both is greater than, less than, exactly equal or both different from unity."""

        names = {'class': 'norm_squared_of_unity', 'positive': 'greater_than_unity', 'negative': 'less_than_unity', 'divider': 'unity'}
        result = self.get_class(self.q1_norm_squared_minus_1, self.q2_norm_squared_minus_1, names, 'dt')
        return result

    def compare(self, eq_2):
        """Compares one set of equivalence classes to anther."""

        pass

    def get_all_classes(self, eq_2=None):
        """Run them all."""
        
        if eq_2 is None:
            eq_classes = [self]
        else:
            eq_classes = [self, eq_2]
            
        for eq_class in eq_classes:
            if 'time' not in eq_class.classes:
                eq_class.time()
            if 'space' not in eq_class.classes:
                eq_class.space()
            if 'causality' not in eq_class.classes:
                eq_class.causality()
            if 'space-times-time' not in eq_class.classes:
                eq_class.space_times_time()
            if 'norm_squared_of_unity' not in eq_class.classes:
                eq_class.norm_squared_of_unity()

    def visualize(self, eq_2=None):
        """Visualize one or two rows of classes with icons for each of the 5 classes."""
        
        self.get_all_classes(eq_2)
        
        if eq_2 is None:
            fig = plt.figure()
            plt.rcParams["figure.figsize"] = [50, 30]
            
            ax1 = fig.add_subplot(3, 5, 1)
            ax1.imshow(self.eq_images['time_' + self.classes['time']])
            plt.axis('off')
            
            ax21 = fig.add_subplot(3, 5, 2)
            ax21.imshow(self.eq_images['space-1_' + self.classes['space-1']])
            plt.axis('off');

            ax22 = fig.add_subplot(3, 5, 7)
            ax22.imshow(self.eq_images['space-2_' + self.classes['space-2']])
            plt.axis('off');

            ax23 = fig.add_subplot(3, 5, 12)
            ax23.imshow(self.eq_images['space-3_' + self.classes['space-3']])
            plt.axis('off');

            ax3 = fig.add_subplot(3, 5, 3)
            ax3.imshow(self.eq_images['causality_' + self.classes['causality']])
            plt.axis('off');

            ax41 = fig.add_subplot(3, 5, 4)
            ax41.imshow(self.eq_images['space-times-time-1_' + self.classes['space-times-time-1']])
            plt.axis('off');

            ax42 = fig.add_subplot(3, 5, 9)
            ax42.imshow(self.eq_images['space-times-time-2_' + self.classes['space-times-time-2']])
            plt.axis('off');

            ax43 = fig.add_subplot(3, 5, 14)
            ax43.imshow(self.eq_images['space-times-time-3_' + self.classes['space-times-time-3']])
            plt.axis('off');

            ax5 = fig.add_subplot(3, 5, 5)
            ax5.imshow(self.eq_images['norm_squared_of_unity_' + self.classes['norm_squared_of_unity']])
            plt.axis('off');

        else:
            fig = plt.figure()
            plt.rcParams["figure.figsize"] = [50, 60]
            
            ax1 = fig.add_subplot(6, 5, 1)
            ax1.imshow(self.eq_images['time_' + self.classes['time']])
            plt.axis('off')
            
            ax21 = fig.add_subplot(6, 5, 2)
            ax21.imshow(self.eq_images['space-1_' + self.classes['space-1']])
            plt.axis('off');

            ax22 = fig.add_subplot(6, 5, 7)
            ax22.imshow(self.eq_images['space-2_' + self.classes['space-2']])
            plt.axis('off');

            ax23 = fig.add_subplot(6, 5, 12)
            ax23.imshow(self.eq_images['space-3_' + self.classes['space-3']])
            plt.axis('off');

            ax3 = fig.add_subplot(6, 5, 3)
            ax3.imshow(self.eq_images['causality_' + self.classes['causality']])
            plt.axis('off');

            ax41 = fig.add_subplot(6, 5, 4)
            ax41.imshow(self.eq_images['space-times-time-1_' + self.classes['space-times-time-1']])
            plt.axis('off');

            ax42 = fig.add_subplot(6, 5, 9)
            ax42.imshow(self.eq_images['space-times-time-2_' + self.classes['space-times-time-2']])
            plt.axis('off');

            ax43 = fig.add_subplot(6, 5, 14)
            ax43.imshow(self.eq_images['space-times-time-3_' + self.classes['space-times-time-3']])
            plt.axis('off');

            ax5 = fig.add_subplot(6, 5, 5)
            ax5.imshow(self.eq_images['norm_squared_of_unity_' + self.classes['norm_squared_of_unity']])
            plt.axis('off');
            

            ax21 = fig.add_subplot(6, 5, 16)
            ax21.imshow(self.eq_images['time_' + eq_2.classes['time']])
            plt.axis('off')
            
            ax221 = fig.add_subplot(6, 5, 17)
            ax221.imshow(self.eq_images['space-1_' + eq_2.classes['space-1']])
            plt.axis('off');

            ax222 = fig.add_subplot(6, 5, 22)
            ax222.imshow(self.eq_images['space-2_' + eq_2.classes['space-2']])
            plt.axis('off');

            ax223 = fig.add_subplot(6, 5, 27)
            ax223.imshow(self.eq_images['space-3_' + eq_2.classes['space-3']])
            plt.axis('off');

            ax23 = fig.add_subplot(6, 5, 18)
            ax23.imshow(self.eq_images['causality_' + eq_2.classes['causality']])
            plt.axis('off');

            ax241 = fig.add_subplot(6, 5, 19)
            ax241.imshow(self.eq_images['space-times-time-1_' + eq_2.classes['space-times-time-1']])
            plt.axis('off');

            ax242 = fig.add_subplot(6, 5, 24)
            ax242.imshow(self.eq_images['space-times-time-2_' + eq_2.classes['space-times-time-2']])
            plt.axis('off');

            ax243 = fig.add_subplot(6, 5, 29)
            ax243.imshow(self.eq_images['space-times-time-3_' + eq_2.classes['space-times-time-3']])
            plt.axis('off');

            ax25 = fig.add_subplot(6, 5, 20)
            ax25.imshow(self.eq_images['norm_squared_of_unity_' + eq_2.classes['norm_squared_of_unity']])
            plt.axis('off');
            
    def __str__(self):
        """Prints all the equivalence relations."""
        
        self.get_all_classes()
        
        class_names = ["time", "space-1", "space-2", "space-3", "causality", 
                       "space-times-time-1", "space-times-time-2", "space-times-time-3", 
                       "norm_squared_of_unity"]
        
        result = "The equivalence classes for this pair of events are as follows...\n"
        
        result += "q1: {}\n".format(QH(self.q1.q4()))
        result += "q2: {}\n".format(QH(self.q2.q4()))
        result += "q1_squared: {}\n".format(QH(self.q1_square.q4()))
        result += "q2_squared: {}\n".format(QH(self.q2_square.q4()))
        result += "q1_norm_squared -1: {}\n".format(QH(self.q1_norm_squared_minus_1.q4()))
        result += "q2_norm_squared -1: {}\n".format(QH(self.q2_norm_squared_minus_1.q4()))
        
        for class_name in class_names:
            result += "{cn:>20}: {c}\n".format(cn=class_name, c=self.classes[class_name])

        return result
    


# In[23]:


class TestEQ(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Q8([1.0, 0, 0, 2.0, 0, 3.0, 0, 4.0])
    q2 = QH([0, 4.0, -3.0, 0])
    eq_11 = EQ(q1, q1)
    eq_12 = EQ(q1, q2)
    
    def test_EQ_assignment(self):
        
        self.assertTrue(self.eq_12.q1.dt.p == 1)
        self.assertTrue(self.eq_12.q1.dt.n == 0)
        self.assertTrue(self.eq_12.q1_square.dt.p == 0)
        self.assertTrue(self.eq_12.q1_square.dt.n == 28)
        self.assertTrue(self.eq_12.q1_norm_squared_minus_1.dt.p == 29)
        self.assertTrue(self.eq_12.q1_norm_squared_minus_1.dt.n == 0)
        self.assertTrue(self.eq_12.q2.dt.p == 0)
        self.assertTrue(self.eq_12.q2.dt.n == 0)
        
    def test_get_class(self):
        """Test all time equivalence classes."""
        names = {'class': 'time', 'positive': 'future', 'negative': 'past', 'divider': 'now'}
        result = self.eq_12.get_class(self.q1, self.q1, names, 'dt')
        self.assertTrue(result == 'future_exact')
    
    def test_time(self):
        """Test all time equivalence classes."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.time() == 'now_exact')
        self.assertTrue(self.eq_12.time() == 'disjoint')
        q1f = QH([4.0, 4.0, 4.0, 4.0])
        q1fe = QH([1.0, 4.0, 4.0, 4.0])
        self.assertTrue(EQ(self.q1, q1f).time() == 'future')
        self.assertTrue(EQ(self.q1, q1fe).time() == 'future_exact')
        q1p = QH([-4.0, 4.0, 4.0, 4.0])
        q1pe = QH([-4.0, 1.0, 2.0, 3.0])
        q1pp = QH([-1.0, 1.0, 2.0, 3.0])
        self.assertTrue(EQ(q1p, q1pp).time() == 'past')
        self.assertTrue(EQ(q1p, q1pe).time() == 'past_exact')
        
    def test_space(self):
        """Test space equivalence class."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.space()[0] == 'here_exact')
        self.assertTrue(eq_zero.space()[1] == 'here_exact')
        self.assertTrue(eq_zero.space()[2] == 'here_exact')
        self.assertTrue(self.eq_11.space()[0] == 'left_exact')
        self.assertTrue(self.eq_11.space()[1] == 'down_exact')
        self.assertTrue(self.eq_11.space()[2] == 'far_exact')
        self.assertTrue(self.eq_12.space()[0] == 'disjoint')
        self.assertTrue(self.eq_12.space()[1] == 'down_exact')
        self.assertTrue(self.eq_12.space()[2] == 'disjoint')
        
        q_sp = Q8([1, 0, 0, 4, 0, 6, 0, 8])
        eq_sp = EQ(self.q1, q_sp)
        self.assertTrue(eq_sp.space()[0] == 'left')
        self.assertTrue(eq_sp.space()[1] == 'down')
        self.assertTrue(eq_sp.space()[2] == 'far')
        
    def test_causality(self):
        """Test all time equivalence classes."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.causality() == 'light-like_exact')
        self.assertTrue(self.eq_12.causality() == 'space-like')
        self.assertTrue(self.eq_11.causality() == 'space-like_exact')
        tl = Q8([4, 0, 0, 0, 0, 0, 0, 0])
        t2 = Q8([5, 0, 0, 3, 0, 0, 0, 0])
        t3 = Q8([5, 0, 3, 0, 1, 0, 0, 0])
        eq_t1_t2 = EQ(tl, t2)
        eq_t1_t3 = EQ(tl, t3)
        self.assertTrue(eq_t1_t2.causality() == 'time-like_exact')
        self.assertTrue(eq_t1_t3.causality() == 'time-like')

    def test_space_times_time(self):
        """Test space equivalence class."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.space_times_time()[0] == 'here-now_exact')
        self.assertTrue(eq_zero.space_times_time()[1] == 'here-now_exact')
        self.assertTrue(eq_zero.space_times_time()[2] == 'here-now_exact')
        self.assertTrue(self.eq_11.space_times_time()[0] == 'future-left_exact')
        self.assertTrue(self.eq_11.space_times_time()[1] == 'future-down_exact')
        self.assertTrue(self.eq_11.space_times_time()[2] == 'future-far_exact')
        self.assertTrue(self.eq_12.space_times_time()[0] == 'disjoint')
        self.assertTrue(self.eq_12.space_times_time()[1] == 'future-down_exact')
        self.assertTrue(self.eq_12.space_times_time()[2] == 'disjoint')

    def test_norm_squared_of_unity(self):
        self.assertTrue(self.eq_11.norm_squared_of_unity() == 'greater_than_unity_exact')
        q_1 = Q8([1, 0, 0, 0, 0, 0, 0, 0])
        q_small = Q8([0.1, 0, 0, 0.2, 0, 0, 0, 0])
        q_tiny = Q8([0.001, 0, 0, 0.002, 0, 0, 0, 0])

        eq_1 = EQ(q_1, q_1)
        eq_q1_small = EQ(q_1, q_small)
        eq_small_small = EQ(q_small, q_small)
        eq_small_tiny = EQ(q_small, q_tiny)
        
        self.assertTrue(eq_1.norm_squared_of_unity() == 'unity_exact')
        self.assertTrue(eq_q1_small.norm_squared_of_unity() == 'disjoint')
        self.assertTrue(eq_small_small.norm_squared_of_unity() == 'less_than_unity_exact')
        self.assertTrue(eq_small_tiny.norm_squared_of_unity() == 'less_than_unity')


# In[24]:


suite = unittest.TestLoader().loadTestsFromModule(TestEQ())
unittest.TextTestRunner().run(suite);


# ## Arrays of Quaternions

# Create a class that can make many, many quaternions.

# In[25]:


class QHArray(QH):
    """A class that can generate many quaternions."""
    
    def __init__(self, q_min=QH([0, 0, 0, 0]), q_max=QH([0, 0, 0, 0]), n_steps=100):
        """Store min, max, and number of step data."""
        self.q_min = q_min
        self.q_max = q_max
        self.n_steps = n_steps
    
    def range(self, q_start, q_delta, n_steps, function=QH.add):
        """Can generate n quaternions"""
        
        functions = {}
        functions["add"] = QH.add
        functions["dif"] = QH.dif
        functions["product"] = QH.product
        
        # To do: figure out the operator used in qtype
        
        q_0 = q_start
        q_0_qtype = q_0.qtype
        self.set_min_max(q_0, first=True)
        yield q_0
        
        for n in range(1, n_steps + 1):
            q_1 = function(q_0, q_delta)
            q_1.qtype = "{q0q}+{n}dQ".format(q0q=q_0_qtype, n=n)
            q_0 = q_1.dupe()
            self.set_min_max(q_1, first=False)
            yield q_1
            
    def set_min_max(self, q1, first=False):
        """Sets the minimum and maximum of a set of quaternions as needed."""
        
        if first:
            self.q_min = q1.dupe()
            self.q_max = q1.dupe()
            
        else:
            if q1.t < self.q_min.t:
                self.q_min.t = q1.t
            elif q1.t > self.q_max.t:
                self.q_max.t = q1.t
                
            if q1.x < self.q_min.x:
                self.q_min.x = q1.x
            elif q1.x > self.q_max.x:
                self.q_max.x = q1.x

            if q1.y < self.q_min.y:
                self.q_min.y = q1.y
            elif q1.y > self.q_max.y:
                self.q_max.y = q1.y
            
            if q1.z < self.q_min.z:
                self.q_min.z = q1.z
            elif q1.z > self.q_max.z:
                self.q_max.z = q1.z
            
    def symbol_sub(self, TXYZ_expression, q1):
        """Given a Symbol expression in terms of T X, Y, and Z, plugs in values for q1.t, q1.x, q1.y, and q1.z"""
        
        new_t = TXYZ_expression.t.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_x = TXYZ_expression.x.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_y = TXYZ_expression.y.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_z = TXYZ_expression.z.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        
        return QH([new_t, new_x, new_y, new_z])


# In[26]:


class TestQHArray(unittest.TestCase):
    """Test array making software."""
    
    t1=QH([1,2,3,4])
    qd=QH([10, .2, .3, 1])
    qha = QHArray()
    
    def test_range(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(len(q_list) == 11)
        self.assertTrue(q_list[10].qtype == "Q+10dQ")
        self.assertTrue(q_list[10].z == 14)
    
    def test_min_max(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(self.qha.q_min.t < 1.01)
        self.assertTrue(self.qha.q_max.t > 100)
        self.assertTrue(self.qha.q_min.x < 2.01)
        self.assertTrue(self.qha.q_max.x > 2.9)
        self.assertTrue(self.qha.q_min.y < 4.01)
        self.assertTrue(self.qha.q_max.y > 5.8)
        self.assertTrue(self.qha.q_min.z < 6.01)
        self.assertTrue(self.qha.q_max.z > 13.9)


# In[27]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHArray())
unittest.TextTestRunner().run(suite);


# ## Array of nparrays

# In[28]:


class QHaArray(QHa):
    """A class that can generate many quaternions."""
    
    def __init__(self, q_min=QHa([0, 0, 0, 0]), q_max=QHa([0, 0, 0, 0]), n_steps=100):
        """Store min, max, and number of step data."""
        self.q_min = q_min
        self.q_max = q_max
        self.n_steps = n_steps
    
    def range(self, q_start, q_delta, n_steps, function=QHa.add):
        """Can generate n quaternions"""
        
        functions = {}
        functions["add"] = QHa.add
        functions["dif"] = QHa.dif
        functions["product"] = QHa.product
        
        # To do: figure out the operator used in qtype
        
        q_0 = q_start
        q_0_qtype = q_0.qtype
        self.set_min_max(q_0, first=True)
        yield q_0
        
        for n in range(1, n_steps + 1):
            q_1 = function(q_0, q_delta)
            q_1.qtype = "{q0q}+{n}dQ".format(q0q=q_0_qtype, n=n)
            q_0 = q_1.dupe()
            self.set_min_max(q_1, first=False)
            yield q_1
            
    def set_min_max(self, q1, first=False):
        """Sets the minimum and maximum of a set of quaternions as needed."""
        
        if first:
            self.q_min = q1.dupe()
            self.q_max = q1.dupe()
            
        else:
            if q1.a[0] < self.q_min.a[0]:
                self.q_min.a[0] = q1.a[0]
            elif q1.a[0] > self.q_max.a[0]:
                self.q_max.a[0] = q1.a[0]
                
            if q1.a[1] < self.q_min.a[1]:
                self.q_min.a[1] = q1.a[1]
            elif q1.a[1] > self.q_max.a[1]:
                self.q_max.a[1] = q1.a[1]

            if q1.a[2] < self.q_min.a[2]:
                self.q_min.a[2] = q1.a[2]
            elif q1.a[2] > self.q_max.a[2]:
                self.q_max.a[2] = q1.a[2]
            
            if q1.a[3] < self.q_min.a[3]:
                self.q_min.a[3] = q1.a[3]
            elif q1.a[3] > self.q_max.a[3]:
                self.q_max.a[3] = q1.a[3]


# In[29]:


class TestQHaArray(unittest.TestCase):
    """Test array making software."""
    
    t1 = QHa([1,2,3,4])
    qd = QHa([10, .2, .3, 1])
    qha = QHaArray()
    
    def test_range(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(len(q_list) == 11)
        self.assertTrue(q_list[10].qtype == "Q+10dQ")
        self.assertTrue(q_list[10].a[3] == 14)
    
    def test_min_max(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(self.qha.q_min.a[0] < 1.01)
        self.assertTrue(self.qha.q_max.a[0] > 100)
        self.assertTrue(self.qha.q_min.a[1] < 2.01)
        self.assertTrue(self.qha.q_max.a[1] > 3.9)
        self.assertTrue(self.qha.q_min.a[2] < 3.01)
        self.assertTrue(self.qha.q_max.a[2] > 4.8)
        self.assertTrue(self.qha.q_min.a[3] < 4.01)
        self.assertTrue(self.qha.q_max.a[3] > 13.9)


# In[30]:


qha = QHArray()
t1=QH([1.0,2.0,3.0,4.0])
qd=QH([10, .2, .3, 1])
for q in qha.range(t1,qd,10):
    print(q)
ql=list(qha.range(t1,qd,10))
print(ql[0])


# In[31]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHaArray())
unittest.TextTestRunner().run(suite);


# An example of making a symbolic calculation.

# In[32]:


T, X, Y, Z = sp.symbols('T X Y Z')
qabstract=QH([T, sp.cos(T), Y + sp.sin(T), Z+T])
print(qabstract)
print(qabstract.t.subs(T,1.0).subs(X,1).subs(Y,2).subs(Z,3))
print(qabstract.x.subs(T,1.0).subs(X,1).subs(Y,2).subs(Z,3))
print(qabstract.y.subs(T,1.0).subs(X,1).subs(Y,2).subs(Z,3))
print(qabstract.z.subs(T,1.0).subs(X,1).subs(Y,2).subs(Z,3))
q2=qha.symbol_sub(qabstract, t1)
print(q2)
for q in qha.range(t1,qd,10):
    print(qha.symbol_sub(qabstract, q))


# ## States - n quaternions that add up to one quaternion

# Any quaternion can be viewed as the sum of n other quaternions. This is common to see in quantum mechanics, whose needs are driving the development of this class and its methods.

# In[33]:


class QHStates(QH):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None):
        
        self.qs = qs
        
        if qs is not None:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q)
        
        return states.rstrip()
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(QHStates(new_states))

    def add(self, ket):
        """Add two states."""
        
        if self.dim != ket.dim:
            oops = "The dimensions are not the same: {} != {}".format(self.dim, ket.dim)
            print(oops)
            return None
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(QHStates(new_states))

    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = QH()
    
        for q in self.qs:
            result = result.add(q)
            
        return result    
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(QHStates(new_states))  
        
    @staticmethod
    def diagonal(q, dim):
        """Make a state dim*dim with q along the 'diagonal'."""
        
        diagonal = []
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q)
                else:
                    diagonal.append(QH().q_0())
        
        return QHStates(diagonal)
        
    @staticmethod
    def product(bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = QHStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = QH().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                if operator.dim == 1:
                    one_diagonal = QHStates().diagonal(operator.qs[0], bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = operator.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = QH().q_0()
                    
                    for b, op in zip(bra.qs, ops): 
                        bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = QHStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = QH().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = QHStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return QHStates(new_states)
    
    @staticmethod
    def Euclidean_product(bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
                    
        return QHStates().product(bra, ket, operator, kind)
    
    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm


# In[34]:


class TestQHStates(unittest.TestCase):
    """Test states."""
    
    q0 = QH().q_0()
    q1 = QH().q_1()
    qi = QH().q_i()
    q0_q1 = QHStates([q0, q1])
    q1_q0 = QHStates([q1, q0])
    q1_qi = QHStates([q1, qi])
    A = QHStates([QH([4,0,0,0]),QH([0,1,0,0])])
    B = QHStates([QH([0,0,1,0]),QH([0,0,0,2]),QH([0,3,0,0])])
    Op = QHStates([QH([3,0,0,0]),QH([0,1,0,0]),QH([0,0,2,0]),QH([0,0,0,3]),QH([2,0,0,0]),QH([0,4,0,0])])
    Op4i = QHStates([QH([0,4,0,0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1].x == -1)
        self.assertTrue(qc1.qs[1].x == 1)
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is QH)
        self.assertTrue(q_01_sum.t == 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].t == 1)
        self.assertTrue(q_0110_add.qs[1].t == 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].t == -1)
        self.assertTrue(q_0110_dif.qs[1].t == 1)
        
    def test_product_AA(self):
        AA = QHStates().product(bra=self.A, ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(QH([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QH([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = QHStates().Euclidean_product(bra=self.A, ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(QH([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QH([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = QHStates().product(bra=self.A, operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QH([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(QH([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(QH([4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = QHStates().Euclidean_product(bra=self.A, operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QH([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(QH([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(QH([12, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = QHStates().product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QH([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QH([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = QHStates().Euclidean_product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QH([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QH([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = QHStates().product(bra=self.A, operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QH([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QH([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = QHStates().Euclidean_product(bra=self.A, operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QH([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QH([-3, 12, 4, 0])))


# In[35]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHStates())
unittest.TextTestRunner().run(suite);


# Repeat this exercise for:
# 
# QHa
# Q8
# Q8a
# 
# by old fashioned cut and paste with minor tweaks (boring).

# In[36]:


class QHaStates(QHa):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None):
        
        self.qs = qs
        
        if qs is not None:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q)
        
        return states.rstrip()
    
    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = QHa()
    
        for q in self.qs:
            result = result.add(q)
            
        return result
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(QHaStates(new_states))
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(QHaStates(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(QHaStates(new_states))  
    
    @staticmethod
    def product(bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = QHaStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = QHa().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                if operator.dim == 1:
                    one_diagonal = QHaStates().diagonal(operator.qs[0], bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = operator.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = QHa().q_0()
                    
                    for b, op in zip(bra.qs, ops): 
                        bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = QHaStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = QHa().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = QHaStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return QHaStates(new_states)

    @staticmethod
    def Euclidean_product(bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
                    
        return QHaStates().product(bra, ket, operator, kind)
    
    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm


# In[37]:


class TestQHaStates(unittest.TestCase):
    """Test states."""
    
    q0 = QHa().q_0()
    q1 = QHa().q_1()
    qi = QHa().q_i()
    q0_q1 = QHaStates([q0, q1])
    q1_q0 = QHaStates([q1, q0])
    q1_qi = QHaStates([q1, qi])
    A = QHaStates([QHa([4,0,0,0]),QHa([0,1,0,0])])
    B = QHaStates([QHa([0,0,1,0]),QHa([0,0,0,2]),QHa([0,3,0,0])])
    Op = QHaStates([QHa([3,0,0,0]),QHa([0,1,0,0]),QHa([0,0,2,0]),QHa([0,0,0,3]),QHa([2,0,0,0]),QHa([0,4,0,0])])
    Op4i = QHaStates([QHa([0,4,0,0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is QHa)
        self.assertTrue(q_01_sum.a[0] == 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].a[0] == 1)
        self.assertTrue(q_0110_add.qs[1].a[0] == 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].a[0] == -1)
        self.assertTrue(q_0110_dif.qs[1].a[0] == 1)
        
    def test_product_AA(self):
        AA = QHaStates().product(bra=self.A, ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(QHa([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QHa([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = QHaStates().Euclidean_product(bra=self.A, ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(QHa([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QHa([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = QHaStates().product(bra=self.A, operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QHa([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(QHa([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(QHa([4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = QHaStates().Euclidean_product(bra=self.A, operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QHa([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(QHa([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(QHa([12, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = QHaStates().product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QHa([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QHa([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = QHaStates().Euclidean_product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QHa([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QHa([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = QHaStates().product(bra=self.A, operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QHa([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QHa([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = QHaStates().Euclidean_product(bra=self.A, operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QHa([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QHa([-3, 12, 4, 0])))


# In[38]:


A = QHaStates([QHa([4,0,0,0]),QHa([0,1,0,0])])
B = QHaStates([QHa([0,0,1,0]),QHa([0,0,0,2]),QHa([0,3,0,0])])
Op = QHaStates([QHa([3,0,0,0]),QHa([0,1,0,0]),QHa([0,0,2,0]),QHa([0,0,0,3]),QHa([2,0,0,0]),QHa([0,4,0,0])])
Op4i = QHaStates([QHa([0,4,0,0])])
print(QHaStates().product(bra=A, operator=Op)) 
print(A.conj()) 
print(QHaStates().Euclidean_product(bra=A, operator=Op)) 


# In[39]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHaStates())
unittest.TextTestRunner().run(suite);


# In[40]:


class Q8States(Q8):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None):
        
        self.qs = qs
        
        if qs is not None:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q)
        
        return states.rstrip()
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(Q8States(new_states))
    
    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = Q8()
    
        for q in self.qs:
            result = result.add(q)
            
        return result
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(Q8States(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(Q8States(new_states))  
    
    @staticmethod
    def product(bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = Q8States().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = Q8().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                if operator.dim == 1:
                    one_diagonal = Q8States().diagonal(operator.qs[0], bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = operator.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = Q8().q_0()
                    
                    for b, op in zip(bra.qs, ops): 
                        bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = Q8States().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = Q8().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = Q8States(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return Q8States(new_states)
    
    @staticmethod
    def Euclidean_product(bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
                    
        return Q8States().product(bra, ket, operator, kind)
    
    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm


# In[41]:


class TestQ8States(unittest.TestCase):
    """Test states."""
    
    q0 = Q8().q_0()
    q1 = Q8().q_1()
    qi = Q8().q_i()
    q0_q1 = Q8States([q0, q1])
    q1_q0 = Q8States([q1, q0])
    q1_qi = Q8States([q1, qi])
    A = Q8States([Q8([4,0,0,0]),Q8([0,1,0,0])])
    B = Q8States([Q8([0,0,1,0]),Q8([0,0,0,2]),Q8([0,3,0,0])])
    Op = Q8States([Q8([3,0,0,0]),Q8([0,1,0,0]),Q8([0,0,2,0]),Q8([0,0,0,3]),Q8([2,0,0,0]),Q8([0,4,0,0])])
    Op4i = Q8States([Q8([0,4,0,0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        print("qc.qs[1]: ", qc.qs[1])
        self.assertTrue(qc.qs[1].dx.n == 1)
        self.assertTrue(qc1.qs[1].dx.p == 1)   
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is Q8)
        self.assertTrue(q_01_sum.dt.p== 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].dt.p== 1)
        self.assertTrue(q_0110_add.qs[1].dt.p== 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].dt.n== 1)
        self.assertTrue(q_0110_dif.qs[1].dt.p== 1)
        
    def test_product_AA(self):
        AA = Q8States().product(bra=self.A, ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(Q8([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = Q8States().Euclidean_product(bra=self.A, ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(Q8([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = Q8States().product(bra=self.A, operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8([4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = Q8States().Euclidean_product(bra=self.A, operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8([12, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = Q8States().product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = Q8States().Euclidean_product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = Q8States().product(bra=self.A, operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = Q8States().Euclidean_product(bra=self.A, operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8([-3, 12, 4, 0])))


# In[42]:


suite = unittest.TestLoader().loadTestsFromModule(TestQ8States())
unittest.TextTestRunner().run(suite);


# In[43]:


class Q8aStates(Q8a):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None):
        
        self.qs = qs
        
        if qs is not None:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q)
        
        return states.rstrip()
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(Q8aStates(new_states))
    
    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = Q8a()
    
        for q in self.qs:
            result = result.add(q)
            
        return result
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(Q8aStates(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(Q8aStates(new_states))  
    
    @staticmethod
    def product(bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = Q8aStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = Q8a().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                if operator.dim == 1:
                    one_diagonal = Q8aStates().diagonal(operator.qs[0], bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = operator.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = Q8a().q_0()
                    
                    for op, k in zip(ops, bra.qs): 
                        bop = bop.add(op.product(k, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = Q8aStates().diagonal(operator.qs[0], ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = Q8a().q_0()
                    
                    for op, k in zip(ops, ket.qs): 
                        ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = Q8aStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return Q8aStates(new_states)
    
    @staticmethod
    def Euclidean_product(bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
                    
        return Q8aStates().product(bra, ket, operator, kind)

    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm


# In[44]:


class TestQ8aStates(unittest.TestCase):
    """Test states."""
    
    q0 = Q8a().q_0()
    q1 = Q8a().q_1()
    qi = Q8a().q_i()
    q0_q1 = Q8aStates([q0, q1])
    q1_q0 = Q8aStates([q1, q0])
    q1_qi = Q8aStates([q1, qi])
    A = Q8aStates([Q8a([4,0,0,0]),Q8a([0,1,0,0])])
    B = Q8aStates([Q8a([0,0,1,0]),Q8a([0,0,0,2]),Q8a([0,3,0,0])])
    Op = Q8aStates([Q8a([3,0,0,0]),Q8a([0,1,0,0]),Q8a([0,0,2,0]),Q8a([0,0,0,3]),Q8a([2,0,0,0]),Q8a([0,4,0,0])])
    Op4i = Q8aStates([Q8a([0,4,0,0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
    
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1].a[3] == 1)
        self.assertTrue(qc1.qs[1].a[2] == 1)
    
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is Q8a)
        self.assertTrue(q_01_sum.a[0]== 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].a[0]== 1)
        self.assertTrue(q_0110_add.qs[1].a[0]== 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].a[1]== 1)
        self.assertTrue(q_0110_dif.qs[1].a[0]== 1)
        
    def test_product_AA(self):
        AA = Q8aStates().product(bra=self.A, ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(Q8a([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8a([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = Q8aStates().Euclidean_product(bra=self.A, ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(Q8a([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8a([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = Q8aStates().product(bra=self.A, operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = Q8aStates().Euclidean_product(bra=self.A, operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([12, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = Q8aStates().product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8a([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = Q8aStates().Euclidean_product(ket=self.B, operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8a([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = Q8aStates().product(bra=self.A, operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8a([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8a([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = Q8aStates().Euclidean_product(bra=self.A, operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8a([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8a([-3, 12, 4, 0])))


# In[45]:


suite = unittest.TestLoader().loadTestsFromModule(TestQ8aStates())
unittest.TextTestRunner().run(suite);

