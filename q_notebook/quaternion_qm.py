
# coding: utf-8

# # Quaternion Quantum Mechanics

# by Doug Sweetser, email to sweetser@alum.mit.edu

# This notebook is being created as a companion to the book "Quantum Mechanics: the Theoretical Minimum" by Susskind and Friedman. Those authors of course never use quaternions as they are a bit player in the crowded field of mathematical tools. I have a different perspective on what makes an equation quantum mechanical. If a conjugate operator is used, then the expression is about quantum mechanics. With quaternions, there is not one conjugate operator, but there are three. First, import the tools to work with quaternions in this notebook.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;')


# Start with the conjugate operator all are familiar with, the one that flips all three imaginary numbers as the Gang of Three:

# In[2]:


t, x, y, z = sp.symbols("t x y z")
R = qt.QH([t, x, y, z])
print(R)
print(R.conj())


# With complex numbers, any function that one can represent in $\mathbb{R}^2$ can just as well be represented with $\mathbb{C}^1$ using $z$ and $z^*$. The same cannot be said using just one quaternion conjugate, there needs to be two others. The idea is not complicated. In the standard conjugate, the real-valued $t$ gets to keep its sign. Why not let other terms have that privilege? If one were to pre- and post-multiply by the imaginary $i$, every term goes back to their original chairs. Both $t$ and $x$ are real at one step of the triple product, and get one minus sign, so stay negative. The $y$ and $z$ get two sign flips, so remain positive about the operation. If one takes the conjugate of this triple product, then $x$ will be the only positive term. I call this the first conjugate.

# In[3]:


Qi, Qj, Qk = qt.QH([0, 1, 0, 0]), qt.QH([0, 0, 1, 0]), qt.QH([0, 0, 0, 1])
first_conj = Qi.product(R.product(Qi)).conj()
print(first_conj)


# The same trick can be done for the second conjugate using a triple product with $j$. The conj() operator has been designed to take the value of 1 or 2 and generate the appropriate conjugate.

# In[4]:


print(R.conj(1))
print(R.conj(2))


# What about the third conjugate operator? Look what happens when we combine the three we have.

# In[5]:


print(R.conj().conj(1).conj(2))


# It is a sign flip a way from doing the job, so flip the signs.

# In[6]:


print(R.conj().conj(1).conj(2).flip_signs())


# Having another conjugate operator is not necessary. Mathematics aspires to be minimalist.

# ## Bracket Notation and Three Identities

# Bracket notation from this quaternion-centric perspective is just a quaternion product where the first term must necessarily be conjugated. I have called this the "Euclidean product". The Euclidean product is not an associative operator unlike the quaternion product. Write out three things in bracket notation that are known to be true.
# 1. $<A|A> \rightarrow A^* A$ is real
# 1. $<A|B> = <B|A>^* \rightarrow A^* B = (B^* A)^*$
# 1. $(<A|+<B|)|C> = <A|C> + <B|C> \rightarrow (A+ B)^*C = A^*C + B^* C$
# 
# Note: taken from page 31 of "Quantum Mechanics, the Theoretical Minimum" by Susskind and Friedman. This will provide the first signs that the odd math of quantum mechanics is the math of Euclidean products.

# In[7]:


a0, A1, A2, A3 = sp.symbols("a0 A1 A2 A3")
b0, B1, B2, B3 = sp.symbols("b0 B1 B2 B3")
c0, C1, C2, C3 = sp.symbols("c0 C1 C2 C3")
A = qt.QH([a0, A1, A2, A3])
B = qt.QH([b0, B1, B2, B3])
C = qt.QH([c0, C1, C2, C3])
print(A.conj().product(A))


# The first term is a real-valued, with the imaginary vector equal to zero. It is important to understand exactly why this is so. A quaternion point in the direction of itself, so that all the anti-symmetric cross terms are equal to zero. The conjugate operator picks out the mirror reflection of the quaternion. The product of an imaginary with its mirror image is an all positive real number and zero for all three imaginary numbers.
# 
# Calculus is the story of neighborhoods near points. There are two classes of changes one can imagine. In the first, a point $A$ goes to $A'$. It could be either slightly bigger or smaller, shown in a slightly bigger or smaller first value. Or the mirror reflection to be slightly off. This would create a non-zero space-times-time 3-vector. 
# 
# Note that the first and second conjugates do not have the "make it all real" property in any obvious way. This is due to the fact that the operations point the 3-vector in a different direction. See what happens when one uses the same operation on the two other conjugate functions:

# In[8]:


print(A.conj(1).product(A))
print(A.conj(2).product(A))


# This result was not what I expected. The square of the norm, $A^* A$, has three zero terms and one non-negative one. The opposite situation occurs for the first and second conjugates: there is one zero, and potentially three non-zero terms. At this point, I don't know what this mean, but will keep the observation in mind.

# Test out the second identity:
# 
# $$(A^* B)^* = (B^*, A)$$

# In[9]:


AB_conj = A.conj().product(B).conj()
BA = B.conj().product(A)
print(AB_conj)
print(BA)
print(AB_conj.dif(BA))


# Despite the fact that quaternions do not commute, the conjugate operator does the job correctly because the angle between the two quaternions does not change. The same cannot be said for the first conjugate:

# In[10]:


AB_conj_1 = A.conj(1).product(B).conj(1)
BA_conj_1 = B.conj(1).product(A)
print(AB_conj_1)
print(BA_conj_1)
print(AB_conj_1.dif(BA_conj_1))


# Again, I don't know what to make of this particular result.

# Now for the third identity about sums.

# In[11]:


A_plus_B_then_C = A.conj().add(B.conj()).product(C)
AC_plus_BC = A.conj().product(C).add(B.conj().product(C))
print(A_plus_B_then_C)
print(AC_plus_BC)
print(AC_plus_BC.dif(AC_plus_BC))


# It is a minor struggle to get the terms to "look" the same, but subtracting one from the other shows they are identical. Does this property hold for the other two conjugates? I think it should, but let's test it:

# In[12]:


A_plus_B_then_C_conj1 = A.conj(1).add(B.conj(1)).product(C)
AC_plus_BC_conj1 = A.conj(1).product(C).add(B.conj(1).product(C))
print(A_plus_B_then_C_conj1)
print(AC_plus_BC_conj1)
print(AC_plus_BC_conj1.dif(AC_plus_BC_conj1))


# In[13]:


A_plus_B_then_C_conj2 = A.conj(2).add(B.conj(2)).product(C)
AC_plus_BC_conj2 = A.conj(2).product(C).add(B.conj(2).product(C))
print(A_plus_B_then_C_conj2)
print(AC_plus_BC_conj2)
print(AC_plus_BC_conj2.dif(AC_plus_BC_conj2))


# This identity does apply to all three conjugate operators.

# ## Quantum States and Quaternion Series

# A quantum state is an n-dimensional vector space. This is fundamentally different from a set of states because certain math relationships are allowed. Vectors can be added to one another, multiplied by complex numbers. One can take the inner product of two vectors.
# 
# A perspective I will explore here is that a (possibly infinite) series of quaternions has the same algebraic properties of Hilbert spaces when one uses the Euclidean product, $A^* B = \sum_{1}^{n} a_n^* b_n$.

# In[37]:


q0 = qt.QH().q_0()
q1 = qt.QH().q_1()

u = qt.QHStates([q1, q0])
d = qt.QHStates([q0, q1])

print("u")
print(u)
print("d")
print(d)


# Calculate $<u|u>$, $<d|d>$ and $<u|d>$:

# In[38]:


uu = u.Euclidean_product(u)
uu_sum = uu.summation()
print("<u|u>")
print(uu)
print("sum")
print(uu_norm)


# In[43]:


dd = d.Euclidean_product(d)
dd_sum = dd.summation()
print("<d|d>")
print(dd)
print("sum: ", dd_sum)


# In[41]:


ud = u.Euclidean_product(d)
ud_sum = ud.summation()
print("<u|d>")
print(ud)
print("sum: ", ud_sum)


# The next pair of states uses $u$ and $d$ like so (TTM, page 41):

# In[48]:


sqrt_2 = qt.QH([sp.sqrt(1/2), 0, 0, 0])

r = qt.QHStates([qt.QH([sp.sqrt(1/2),0,0,0]), qt.QH([sp.sqrt(1/2),0,0,0])])
L = qt.QHStates([qt.QH([sp.sqrt(1/2),0,0,0]), qt.QH([- sp.sqrt(1/2),0,0,0])])

print("r")
print(r)
print("L")
print(L)


# In[54]:


rr = r.Euclidean_product(r)
rr_sum = rr.summation()
print("<r|r>")
print(rr)
print("sum")
print(rr_sum)
print()
LL = L.Euclidean_product(L)
LL_sum = LL.summation()
print("<L|L>")
print(LL)
print("sum")
print(LL_sum)
print()
rL = r.Euclidean_product(L)
rL_sum = rL.summation()
print("<r|L>")
print(rL)
print("sum")
print(rL_sum)


# Orthonormal again, as hoped for.

# The final calculation for chapter 2 is like the one for $r$ and $L$ except one uses an arbitrarily chosen imaginary value - it could point any direction in 3D space - like so:

# In[52]:


i = qt.QHStates([qt.QH([sp.sqrt(1/2),0,0,0]), qt.QH([0, sp.sqrt(1/2), 0, 0])])
o = qt.QHStates([qt.QH([sp.sqrt(1/2),0,0,0]), qt.QH([0, - sp.sqrt(1/2), 0, 0])])

print("i")
print(i)
print("o")
print(o)


# In[53]:


ii = i.Euclidean_product(i)
ii_sum = ii.summation()
print("<i|i>")
print(ii)
print("sum")
print(ii_sum)
print()
oo = o.Euclidean_product(o)
oo_sum = oo.summation()
print("<o|o>")
print(oo)
print("sum")
print(oo_sum)
print()
io = i.Euclidean_product(o)
io_sum = io.summation()
print("<i|o>")
print(io)
print("sum")
print(io_sum)

