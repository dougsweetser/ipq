
# coding: utf-8

# # Quaternion Series Quantum Mechanics: Lectures 1 and 2

# by Doug Sweetser, email to sweetser@alum.mit.edu

# This notebook is being created as a companion to the book "Quantum Mechanics: the Theoretical Minimum" by Susskind and Friedman (QM:TTM for short). Those authors of course never use quaternions as they are a bit player in the crowded field of mathematical tools. Nature has used one accounting system since the beginning of space-time, so I will be a jerk in the name of consistency. This leads to a different perspective on what makes an equation quantum mechanical. If a conjugate operator is used, then the expression is about quantum mechanics. It is odd to have such a brief assertion given the complexity of the subject, but that make the hypothesis fun - and testable by seeing if anything in the book cannot be done with quaternions and their conjugates. Import the tools to work with quaternions in this notebook.

# In[2]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;')


# ## Lecture 1: Systems and Experiments

# ### Bracket Notation and Three Identities

# Bracket notation from this quaternion-centric perspective is just a quaternion product where the first term must necessarily be conjugated. I have called this the "Euclidean product". The quaternion product is associative but the Euclidean product is not ($(A^* B)^* C \ne A^* (B^* C)$ although their norms are equal). Write out three things in bracket notation that are known to be true about inner products(QM:TTH, p. 31).
# 1. $<A|A> \rightarrow A^* A$ is real
# 1. $<A|B> = <B|A>^* \rightarrow A^* B = (B^* A)^*$
# 1. $(<A|+<B|)|C> = <A|C> + <B|C> \rightarrow (A+ B)^*C = A^*C + B^* C$
# 
# This may provide the first signs that the odd math of quantum mechanics is the math of Euclidean products of quaternions.
# 
# So, is $A^* A$ real? Yes and no.

# In[3]:


a0, A1, A2, A3 = sp.symbols("a0 A1 A2 A3")
b0, B1, B2, B3 = sp.symbols("b0 B1 B2 B3")
c0, C1, C2, C3 = sp.symbols("c0 C1 C2 C3")
A = qt.QH([a0, A1, A2, A3], qtype="A")
B = qt.QH([b0, B1, B2, B3], qtype="B")
C = qt.QH([c0, C1, C2, C3], qtype="C")
display(A.conj().product(A).t)
display(A.conj().product(A).x)
display(A.conj().product(A).y)
display(A.conj().product(A).z)


# The first term is a real-valued, with the 3-imaginary vector equal to zero. I think it is bad practice to just pretend the three zeros are *not there in any way*. One can make an equivalence relation between quaternions of the form $(\mathbb{R}, 0, 0, 0)$ and the real numbers. The real numbers are a subgroup of quaternions, and never the other way around.
# 
# It is important to understand exactly why the three imaginary terms are zero. It is too common for people to say "it's the norm" and give the subject no thought. No thought means no insights. A quaternion points in the direction of itself, so all the anti-symmetric cross terms are equal to zero. The conjugate operator picks out the mirror reflection of the imaginary terms. The product of an imaginary with its mirror image is an all positive real number and zero for all three imaginary numbers.
# 
# Calculus is the story of neighborhoods near points. There are two broad classes of changes one can imagine for a norm. In the first, a point $A$ goes to $A'$. It could be either slightly bigger or smaller, shown in a slightly bigger or smaller first value. Or the mirror reflection to be slightly off. This would create a non-zero space-times-time 3-vector. Everyone accepts that a norm can get larger or smaller, it is a "size" thing. But a change in direction will lead to imaginary terms that can either commute, anti-commute, or be a mixture of both. This possibility makes this view of a quaternion norm sound richer.

# Test out the second identity:
# 
# $$(A^* B)^* = (B^*, A)$$

# In[4]:


AB_conj = A.Euclidean_product(B)
BA = B.Euclidean_product(A).conj()
print("(A* B)* = {}".format(AB_conj))
print("B* A    = {}".format(BA))
print("(A* B)* - B* A = {}".format(AB_conj.dif(BA)))


# Note on notation: someone pointed out that is *absolutely all calculations start and end with quaternions*, then it is easy to feel lost - this quaternion looks like that one. The string at the end that I call a "qtype" represents all the steps that went into a calculation. The last qtype above reads: A*xB-B*xA* which hopefully is clear in this contex.

# Despite the fact that quaternions do not commute, the conjugate operator does the job correctly because the angle between the two quaternions does not change.

# Now for the third identity about sums.

# In[5]:


A_plus_B_then_C = A.conj().add(B.conj()).product(C).expand_q()
AC_plus_BC = A.conj().product(C).add(B.conj().product(C)).expand_q()
print("(A+B)* C:  {}\n".format(A_plus_B_then_C))
print("A*C + B*C: {}\n".format(AC_plus_BC))
print("(A+B)* C - (A*C + B*C): {}".format(A_plus_B_then_C.dif(AC_plus_BC)))


# Subtracting one from the other shows they are identical.
# 
# There are many more algebraic relationships known for Hilbert spaces such as the triangle inequality and the Schwarz inequality which is the basis of the uncertainty principle. These all work for the [Euclidean product with quaternions](https://dougsweetser.github.io/Q/QM/bracket_notation/).

# ## Lecture 2: Quantum States

# ### Quaternion Series as Quantum States

# A quantum state is an n-dimensional vector space. This is fundamentally different from a set of states because certain math relationships are allowed. Vectors can be added to one another, multiplied by complex numbers. One can take the inner product of two vectors. Most important calculations involve taking the inner product.
# 
# A perspective I will explore here is that a (possibly infinite) series of quaternions has the same algebraic properties of Hilbert spaces when one uses the Euclidean product, $A^* B = \sum_{1}^{n} a_n^* b_n$

# ![AxB.png](images/AxB.png)

# This only works if the length of the series for **A** is exactly equal to that of **B**. Whatever can be done with a quaternion can be done with its series representation. Unlike vectors that can either be be a row or a column, quaternion series only have a length. Let's just do one calculation, < A | A >:

# In[6]:


A = qt.QHStates([qt.QH([0,1,2,3]), qt.QH([1,2,1,2])])
AA = A.Euclidean_product('bra', ket=A)
AA.print_states("<A|A>")


# A little calculation in the head should show this works as expected - except one is not used to seeing quaternion series in action.

# The first system analyzed has but 2 states, keeping things simple. The first pair of states are likewise so simple they are orthonormal to a casual observer.

# In[7]:


q0, q1, qi, qj, qk = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_i(), qt.QH().q_j(), qt.QH().q_k()

u = qt.QHStates([q1, q0])
d = qt.QHStates([q0, q1])

u.print_states("u", True)
d.print_states("d")


# Calculate $<u|u>$, $<d|d>$ and $<u|d>$:

# In[10]:


u.Euclidean_product('bra', ket=u).print_states("<u|u>")


# In[9]:


d.Euclidean_product('bra', ket=d).print_states("<d|d>")


# In[12]:


u.Euclidean_product('bra', ket=d).print_states("<u|d>")


# The next pair of states is constructed from the first pair, $u$ and $d$ like so (QM:TTM, page 41):

# In[15]:


sqrt_2op = qt.QHStates([qt.QH([sp.sqrt(1/2), 0, 0, 0])])

u2 = u.Euclidean_product('ket', operator=sqrt_2op)
d2 = d.Euclidean_product('ket', operator=sqrt_2op)

r = u2.add(d2)
L = u2.dif(d2)

r.print_states("r", True)
L.print_states("L")


# In[16]:


r.Euclidean_product('bra', ket=r).print_states("<r|r>", True)
L.Euclidean_product('bra', ket=L).print_states("<L|L>", True)
r.Euclidean_product('bra', ket=L).print_states("<r|L>", True)


# The final calculation for chapter 2 is like the one for $r$ and $L$ except one uses an arbitrarily chosen imaginary value - it could point any direction in 3D space - like so:

# In[20]:


i_op = qt.QHStates([q1, q0, q0, qi])

i = r.Euclidean_product('ket', operator=i_op)
o = L.Euclidean_product('ket', operator=i_op)

i.print_states("i", True)
o.print_states("o")


# In[22]:


i.Euclidean_product('bra', ket=i).print_states("<i|i>", True)
o.Euclidean_product('bra', ket=o).print_states("<o|o>", True)
i.Euclidean_product('bra', ket=o).print_states("<i|o>")


# Notice how long the qtypes have gotten (the strings that keep a record of all the manipulations done to a quaternion). The initial state was just a zero and a one, but that had to get added to another and normalized, then multiplied by a factor of $i$ and combined again.

# Orthonormal again, as hoped for.

# Is the quaternion series approach a faithful representation of these 6 states? On page 43-44, there are 8 products that all add up to one half. See if this works out...

# In[26]:


ou = o.Euclidean_product('bra', ket=u)
uo = i.Euclidean_product('bra', ket=o)
print("ouuo sum:\n", ou.product('bra', ket=uo).summation(), "\n")
od = o.Euclidean_product('bra', ket=d)
do = d.Euclidean_product('bra', ket=o)
print("oddo sum:\n", od.product('bra', ket=do).summation(), "\n")
iu = i.Euclidean_product('bra', ket=u)
ui = u.Euclidean_product('bra', ket=i)
print("iuui sum:\n", iu.product('bra', ket=ui).summation(), "\n")
id = i.Euclidean_product('bra', ket=d)
di = d.Euclidean_product('bra', ket=i)
print("iddi sum:\n", id.product('bra', ket=di).summation())


# In[28]:


Or = o.Euclidean_product('bra', ket=r)
ro = r.Euclidean_product('bra', ket=o)
print("orro:\n", Or.product('bra', ket=ro).summation(), "\n")
oL = o.Euclidean_product('bra', ket=L)
Lo = L.Euclidean_product('bra', ket=o)
print("oLLo:\n", oL.product('bra', ket=Lo).summation(), "\n")
ir = i.Euclidean_product('bra', ket=r)
ri = r.Euclidean_product('bra', ket=i)
print("irri:\n", ir.product('bra', ket=ri).summation(), "\n")
iL = i.Euclidean_product('bra', ket=L)
Li = L.Euclidean_product('bra', ket=i)
print("iLLi:\n", iL.product('bra', ket=Li).summation())


# There is an important technical detail in this calculation I should point out. In the <bra|ket> form, the bra gets conjugated. Notice though that if one does two of these, < i | L >< L | i >, then there has to be a product formed between the two brackets. In practice, < i | L >* < L | i > gives the wrong result:

# In[29]:


print("iL*Li:\n", iL.Euclidean_product('bra', ket=Li).summation())

