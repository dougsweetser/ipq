
# coding: utf-8

# # Simple Rotations

# In this iPython notebook, the one thing quaternions are known for - doing 3D spatial rotations - will be examined. When working with quaternions, it is not possible to work in just three spatial dimensions. There is always a fourth dimension, namely time. So we are thinking about events in space-time, not just events.
# 
# Create a couple of quaternions to play with.

# In[9]:

from Q_tool_devo import Qq;
u=Qq([1,2,3,4])
r=Qq([5,6,7,8])


# We are using the Qq class that places these 4 numbers in 8 slots like so:

# In[3]:

print(u)
print(r)


# If you are unfamiliar with this notation, the $I^2 = -1,\, i^3=-i,\, j^3=-j,\, k^3=-k$. Only positive numbers are used, with additive inverse put in these placeholders.
# 
# To do a rotaion, one needs to pre-multiply by a quaternion (using u for this role), then post-multiply by the inverse. By doing so, the norm of r will not change because quaternions are a normed division algebra.

# In[4]:

rrotated=u.triple_product(r, u.invert())


# Should we expect the first term to change? Look into the triple product first term where I use Capital variable for 3-vectors to simplify the presentation:
# 
# $$\begin{align}(u, W)&(r, R)(u, -W)/(u^2 + W \cdot W)\\
# &= (u r - W \cdot R, u R + W r + W \times R)(u, -W)/(u^2 + W \cdot W)\\
# &=(u^2 r - uW \cdot R + u W \cdot R + W \cdot W r - W \cdot W \times R, ...)/(u^2 + W \cdot W)\\
# &=\left(\frac{u^2 + W \cdot W}{u^2 + W \cdot W}r,...\right) = (r, ...)\end{align}$$
# Now look at the rotated event.

# In[5]:

print(rrotated)
print(rrotated.reduce())


# The first term does change! At least in the non-reduced Qq format, there is a change because it is composed of the positive and negative terms we saw in the algebra problem. For example, there is the vector identity W.WxR=0. The cross product makes a vector that is 90 degrees to both W and R. The dot product of that cross product with W is zero because nothing is in the direction of W anymore. This shows up algebraically because the 6 terms of the cross product have three positive terms and three negative terms that exactly cancel when dotted to W. But the values remain in the $I^0$ and $I^2$ terms until Qq is reduced.
# 
# When the Qq is reduced, it ends up being a 5 as expected. This may be of interest because we keep more information about the change with the eight positions to fill in the Qq representation (none of which are empty after the rotation).

# We expect the norms to be identical in the reduce form:

# In[6]:

print(r.norm())
print(rrotated.norm())
print(rrotated.norm().reduce())


# If squared, the reduced interval should be the same too:

# In[8]:

print(r.square().norm())
print(rrotated.square().norm())
print(rrotated.square().norm().reduce())


# But what should we make of these non-reduced calculations? Here is my speculation. In classical physics, one always, always, always uses the reduced form of a Qq quaternion measurement. Classical physics involves one thing doing something. Physics gets odd when dealing with relativistic quantum feild theory. That is a rare sport played only when a one packet of protons collides with another inside an atom smasher. In those entirely odd situations, one must start thinking about multiple particles because we cannot know what happened, there is too much energy around, so we sum over all possible histories.
# 
# It is simple to move an event to another place in space-time the same distance from the origin. Because it is a transient event, it feels fleeting, which it should.

# In[ ]:



