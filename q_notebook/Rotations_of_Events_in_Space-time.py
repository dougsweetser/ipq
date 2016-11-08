
# coding: utf-8

# # Simple Rotations

# Author: Doug <sweetser@alum.mit.edu>

# A deep leason from special relativity is that all measurements start as events in space-time. A measurment such as the length of a rod would appear to only involve space. Yet one needs a signal from both ends that gets received by the observer. And who is the observer? Every other system in the Universe is allowed. Since observers can be in very different states, that means the raw data these different observers collect from the same signals can be different. Special relaitivty helps determine what can be agreed upon, namely the interval for two inertial reference frame observers.

# ## Spatial Rotations

# In this iPython notebook, the one thing quaternions are known for - doing 3D spatial rotations - will be examined. When working with quaternions, it is not possible to work in just three spatial dimensions. There is always a fourth dimension, namely time. Even when thinking about spatial rotations, one must work with events in space-time.
# 
# Create a couple of quaternions to play with.

# In[11]:

from Q_tool_devo import Qq;
U=Qq([1,2,3,4])
R=Qq([5,6,7,8])


# Use the Qq class that places these 4 numbers in 8 slots like so:

# In[2]:

print(U)
print(R)


# If you are unfamiliar with this notation, the $I^2 = -1,\, i^3=-i,\, j^3=-j,\, k^3=-k$. Only positive numbers are used, with additive inverse put in these placeholders.
# 
# To do a rotation, one needs to pre-multiply by a quaternion (using u for this role), then post-multiply by the inverse. By doing so, the norm of r will not change because quaternions are a normed division algebra.

# In[3]:

R_rotated=U.triple_product(R, U.invert())


# Should we expect the first term to change? Look into the triple product first term where I use Capital variable for 3-vectors to simplify the presentation:
# 
# $$\begin{align}(u, W)&(t, R)(u, -W)/(u^2 + W \cdot W)\\
# &= (u t - W \cdot R, u R + W t + W \times R)(u, -W)/(u^2 + W \cdot W)\\
# &=(u^2 t - uW \cdot R + u W \cdot R + W \cdot W t - W \cdot W \times R, ...)/(u^2 + W \cdot W)\\
# &=\left(\frac{u^2 + W \cdot W}{u^2 + W \cdot W}t,...\right) = (t, ...)\end{align}$$
# 
# Now look at the rotated event.

# In[4]:

print(R_rotated)
print(R_rotated.reduce())


# The first term does change! At least in the non-reduced Qq format, there is a change because it is composed of the positive and negative terms we saw in the algebra problem. For example, there is the vector identity W.WxR=0. The cross product makes a vector that is 90 degrees to both W and R. The dot product of that cross product with W is zero because nothing is in the direction of W anymore. This shows up algebraically because the 6 terms of the cross product have three positive terms and three negative terms that exactly cancel when dotted to W. But the values remain in the $I^0$ and $I^2$ terms until Qq is reduced.
# 
# When the Qq is reduced, it ends up being a 5 as expected. This may be of interest because we keep more information about the change with the eight positions to fill in the Qq representation (none of which are empty after the rotation).

# We expect the norms to be identical in the reduce form:

# In[5]:

print(R.norm())
print(R_rotated.norm())
print(R_rotated.norm().reduce())


# If squared, the reduced interval should be the same too:

# In[6]:

print(R.square().reduce())
print(R_rotated.square())
print(R_rotated.square().reduce())


# But what should we make of these non-reduced calculations? Here is my speculation. In classical physics, one always, always, always uses the reduced form of a Qq quaternion measurement. Classical physics involves one thing doing something. Physics gets odd when dealing with relativistic quantum feild theory. That is a rare sport played only when a one packet of protons collides with another inside an atom smasher. In those entirely odd situations, one must start thinking about multiple particles because we cannot know what happened, there is too much energy around, so we sum over all possible histories.
# 
# It is simple to move an event to another place in space-time the same distance from the origin. Because it is a transient event, it feels fleeting, which it should.

# ## Rotations in Time

# A rotation in time is commonly called a boost. The idea is that one gets a boost in speed, and that will change measurements of both time and distance. If one rushes toward the source of a signal, both the measurement of time and distance will get shorter in a way that keeps the interval the same.

# In[7]:

R_boosted=R.boost(0.01,0.02, 0.003)
print("boosted: {}".format(R_boosted.reduce()))
print(R.square().reduce())
print(R_boosted.square())
print(R_boosted.square().reduce())


# The reduced interval is $124 \,I^2$, whether boosted or not. The norm will shrink because all the number are a little smaller, no longer quite (5, 6, 7, 8).

# In[8]:

print(R.norm().reduce())
print(R_boosted.norm())
print(R_boosted.norm().reduce())


# ## Rotations in Space and Time

# Quaternions are just numbers. This makes combining transformations trivial. A measurement can be rotated and boosted. The only thing that should be unchanged is the interval:

# In[9]:

R_rotated_and_boosted = R_rotated.boost(0.01,0.02, 0.003)
print("rotated and boosted: {}".format(R_rotated_and_boosted.reduce()))
print(R.square().reduce())
print(R_rotated_and_boosted.square())
print(R_rotated_and_boosted.square().reduce())


# Because of the rotation, the z value was larger. It is a safe bet that the norm turns out to be smaller as happened before:

# In[10]:

print(R.norm().reduce())
print(R_rotated_and_boosted.norm())
print(R_rotated_and_boosted.norm().reduce())


# ## Ratios at Work

# An angle is a ratio, this side over that side. The velocity used in a boost is a ratio of distance over time. Say we have a reference observer who measures the interval between two events. If another observer see the same events, but was standing on his head, we migh expect the headstand to change how the crazy observer places his numbers into the three spatial slots. Yet the two observers should agree to the inteval, and they do.
# 
# The same happens if there is an observer travelling at a constant velocity relative to the referene observer. It is not surprizing that the math machinery is different because a spatial rotation is different from moving along at certain velocity.
# 
# Combining the spatial rotations and boosts can be done, creating messy results, except for the interval that remains the same.
