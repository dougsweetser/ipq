
# coding: utf-8

# # A Padantic Introduction to the Quaternion Gravity Proposal

# The quaternion gravity proposal is the imaginary twin of special relativity. Two observers record the difference between two events. When the difference written as quaternion is squared, if the real interval is identical and thus the imaginary parts hereby called space-times-time must have difference, this is the domain of special relativity. Special relativity is viewed as an equivalence class for the real part which uses a Lorentz transformation to go between the two. The space-times-time can be used to determine exactly the relative motion of the two observers.

# If the space-times-time values are identical, the real interval is different. This is what happens seamingly by chance in general relativity for the Schwarzschild solution in Schwarzschild coordinates outside a static, uncharged, non-rotating spherically-symmetric gravitational source mass. In the quaternion gravity proposal, observers that agree on the three space-times-time values is the equivalence class for gravity. Since this is an eqivalence class, there is a transformation that can be constructed between the two.

# In this notebook/blog, I will go through some detail to demonstrate these two equivalence classes.

# ## Calculation Tools
# 
# In the past, I used Mathematica to confirm my algebra was correct (it has caught errors in the past). The software used to be crazy expensive (~\$2k), but has come down in price (\$300). Still, I would prefer to use an open source project. This motivated me to look into the iPython Notebook. Since I am a fan of the Python language, I took a closer look. I am quite impressed at the current state of affairs. It looked easy to mix and match real python code with text and images. But how to get iPython Notebook up and running? Anaconda.org is a commercial company that provides a free python environment. They are trying to get by through paid support and cloud services for companies. I liked that after installing anaconda, running "jupyter notebook" just worked. Jupyter is the notebook that then attaches to a variety of math tools of your choosing. This blog is being drafted as an iPython Notebook.

# I decided to create my own tools for quaternions from scratch. Why? This is the opening few paragraphs of my "Q_tool_devo" notebook:

#     In this notebook, tools for working with quaternions for physics issues are developed. 
#     The class Qh treat quaternions as Hamilton would have done: as a 4-vector over the real numbers.

#     In physics, group theory plays a central role in the fundamental forces of Nature via 
#     the standard model. The  gauge symmetry U(1) a unit circle in the complex plane leads 
#     to electric charge conservation. The unit quaternions SU(2) is the symmetry needed for 
#     the weak force which leads to beta decay. The group SU(3) is the symmetry of the strong
#     force that keeps a nucleus together.

#     The class Qq was written in the hope that group theory would be written in first, not
#     added as needed later. I call these "space-time numbers". The problem with such an
#     approach is that one does not use the mathematical field of real numbers. Instead one
#     relies on the set of positive reals. In some ways, this is like reverse engineering some
#     basic computer science. Libraries written in C have a notion of a signed versus unsigned
#     integer. The signed integer behaves like the familiar integers. The unsigned integer is
#     like the positive integers. The difference between the two is whether there is a
#     placeholder for the sign or not. All floats are signed. The modulo operations that work
#     for unsigned integers does not work for floats.

#      Test driven development was used. The same tests for class Qh were used for Qq.  Either
#      class can be used to study quaternions in physics.

# Here is a list of the functions that were written for the Qq classes:

#     abs_of_q, abs_of_vector, add, all_products, anti_commuting_products, boost,
#     commuting_products, conj, dif, divide_by, g_shift, invert, norm, norm_of_vector, 
#     product, q4, q_one, q_zero, reduce, rotate, square, and triple_product.

# Like all long list, this makes for dull prose. I had a little fun breaking up the multiplication product into the commuting and anticommuting parts.

# The most unusual method is "reduce". The Qq class uses 8 positive numbers to represent a quaternion. Any number can be represented an inifinite number of ways, so long as the difference between the positive and negative number remains the same. There is however only one reduced form for a number. That will have either the positive or its additive inverse set to zero (both can be zero also). It is mildly amusing to see a complicated calculation that fills up all eight slots that after the reduce step ends up with precisely the same result as appears from the Qh or Hamilton quaternion class that uses real numbers. It feels uncomfortable to me to see these eight numbers since it is not my experience with calculations. Numbers in Nature do deeply odd things (think of the great boson/fermion divide in how states should be filled). 

# I program using a method called Test Driven Development. This means that all methods get a test so one knows each piece is working. That is critcal with programs since one typo will means a tool does not work as expected. The same tests were applied to the Hamilton Qq class as the Quaternion Group Q<sub>8</sub> class Qq. The reduced form of all Qq calculations are the same as the Qh class.

# If you have any interst it playing with the iPython notebook, feel free to clone it:
# 
# > git clone https://github.com/dougsweetser/ipq

# ## Equivalence classes
# 
# An equivalence class is part of set theory. Take a big set, carve it up into subsets, and a subset is an equivalence class. One uses an equivalence relation to determine if something is in a subset. As usual, you can [read more about that on wikipedia](https://en.wikipedia.org/wiki/Equivalence_class).

# Start simple with the future equivalence class $[f]$. To be a member, all that an event needs is a positive measure of time.

# $$[f] = \{ q \in Q \,|\, f \sim q \;\rm{if} \; \rm{Re}(q) > 0 \}$$

# One can define an exact future class $[f_e]$ where two points are in the future the exact same amount:

# $$[f_e] = \{ q \in Q \,|\, f_e \sim q \;\rm{if} \; \rm{Re}(q) > 0 \; \rm{and} \; \rm{Re}(q) = \rm{Re}(f_e)\}$$

# One question can be asked of a pair of quaternions: are they both in the future equivalence class, and if so, are they exactly equal to one another? This type of question was particularly easy to ask of with the Qq class in the reduced form. They would both be positive in the future if they had a non-zero time value. In the case that both were in the future, then one could ask futher if the values were the same, up to a defined rounding error. The computer code felt like it was doing basic set theory: a pair of numbers was in or out. No inequalities were needed.

# The two classes are easy enough to graph:

# ![sdf](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/time_future_future_exact.png)

# Figuring out if a pair of events are both in the past works the same. This time one looks to see if they both have non-zero values in the additive invserse (aka negative) time slot of the Qq class. The same function was used, but telling the function to look at the additive inverses.

# One can also see if both numbers are neither positive or negative. That only happens if the value of time is zero, or now. If so, the pair events gets marked as now exact.

# Plucking events at random, the most common situation is that a pair of events would be disjoint: one in the future, the ohter in the past. This was the default case that resulted after all the other situations were investigated.

# The 6 questions - are you both positive, if so, exact, negative, if so exact, both zeroes or disjoint - can be asked for the three spatial dimensions: left versus right, up versus down, and near versus far. All use the same function to figure out which is the winning equivalence class (although disjoint is not an equivance class).

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/space_classes.png)

# The equivalence class EQ is fed 2 quaternions. It reduces these two to deal with the future/past, left/right, up/down, and near/far equivalance classes. All events in space-time map into four equivalence classes as one might expect for events in space-time.

# There are two more general classes, one exeptionally narrow, the other the most common of all. The narrow class is when both are zero, the *now* class for time, and the *here* class for space.

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/time_now_exact.png)

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/space_here_exact.png)

# Observers have all four of these exact matches since they are *here-now*.

# The most common situation for a pair of events is that they are disjoint. As usual, there are four ways to be disjoint:

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/time_disjoint.png)

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/space_disjoint.png)

# Physics is an observational science. Every pair of events ever belongs in four of these equivalence classes or the disjoint classes. Every combination of these classes is out there in the event library of the Universe. This is the raw data of events. A problem with the raw data is that no observer can ever stay at one here-now. How do we deal with transient number?

# ## Ever Changing Events, Fixed Differences

# ## Special Relativity as the Square of a Delta Quaternion

# My proposal for special relativity works with the square of a quaternion. This must be done because the Lorentz invariant interval for inertial obserrvers is time squared minus space squared. The expression mathematical is nearly idential to telling the past from the future:

# $$[df] = \{ dq \in Q \,|\, df \sim dq \;\rm{if} \; \rm{Re}(dq^2) > 0 \}$$

# The only difference is the square and the use of fixed difference quaternions.

# ![](https://raw.githubusercontent.com/dougsweetser/ipq/master/q_notebook/images/eq_classes/causality_time-like_time-like_exact.png)

# Of course it might be the case that the events were both space-like separated because the first term of the square was negative:

# ![](images/eq_classes/causality_space-like_space-like_exact.png)
# 
# 

# In[ ]:




# In[ ]:



