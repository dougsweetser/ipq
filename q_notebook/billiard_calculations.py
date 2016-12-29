
# coding: utf-8

# # Billiards With Space-time Numbers

# The goal of this iPython notebook is to become familiar with using space-time numbers to describe events. This will be done for three different observers. The first case will cover the fact that the observers happen to be at different locations. How does one handle different ways to represent the numbers used to characterize events?

# One observer will be set in constant motion. We will work out the equivalence classes that cover observers in motion. The final case will look at equivalence classes that may happen due to gravity.

# Here is an animation of a mini billiard shot.

# ![](images/Billiards/billiard_video.gif)

# Cue ball hits the 8 ball, then into the corner pocket it goes. Observer A is yellow, our proverbial reference observer. I promise to do nothing with her ever. Observer B in pink is at a slightly different location, but still watching from the tabletop. Eventually, he will be set into constant motion. Then we can see about what Observers agree and disagree about. Observer C is in purple and at the end of a pipe cleaner above the tabletop. His observations will be ever-so-slightly different from Observer A and that will be investigated.

# A number of big simplifications will be done for this analysis. All but two frames will be used.

# ![](images/Billiards/shot_12.png)

# Get rid of the green felt. In its place, put some graph paper. Add a few markers to make any measurement more precise.

# ![](images/Billiards/shot_12_no_felt.png)

# The image was then printed out so a precise dial calliper could be used to make measurements. Notice that observer A is ~2.5 squares to the left and 3+ squares below the 8 ball in the first frame.

# ![](images/Billiards/ball_printout.jpg)

# Can the time be measured precisely? In this case, I will use the frames of the gif animation as a proxy for measuring time. I used the command "convert billiard_video.gif Frames/billiards_1%02d.png" to make make individual frames from the gif. The two frames are 147 and 158. The speed of the fastest cue break is over 30 miles per hour, or as a diminsionless relativistic speed is 4.5x10<sup>-8</sup>. If small number are used for the differences in space, then the difference between time should be scaled to be in the ten billion range. So that is what I did: call the first time 1,470,000,000 and the second one 1,580,000,000. The ball is then moving around 20mph. I could have found out the frames per second, and calculated the correct speed from there. The three observers do not need to coordinate to figure out the same origin in time, so I chose B and C to start a billion and two billion earlier respectively.

# This explains how I got numbers related to an 8 ball moving on a table. Now to calculate the in

# In[1]:

get_ipython().run_cell_magic('capture', '', 'import Q_tool_devo as qtd;\nAq1=qtd.Qq([1470000000,0,1.1421,0,1.4220,0,0,0])\nAq2=qtd.Qq([1580000000,0,4.2966,0,0,0.3643,0,0])')


# In[2]:

q_scale = qtd.Qq([2.2119,0,0,0,0,0,0,0], qtype="S")
Aq1s=Aq1.product(q_scale)
Aq2s=Aq2.product(q_scale)
print(Aq1s)
print(Aq2s)


# When scaled, the expected values are seen, the x value at around 2.5, the y value above 3 and zero for z.

# Calculate the delta quaternion between events one and two:

# In[3]:

Adq=Aq2s.dif(Aq1s).reduce()
print(Adq)


# The difference is nearly 7 in the x<sub>1</sub> direction, and 4 in the j<sub>3</sub>, which if real numbers were being used would be the positive x and negative y. The qtype "QxQ-QxQ.reduce" shows that both initial components were multiplied by a scalar value, then the difference taken.

# In[4]:

Adq2=Adq.square()
print(Adq2)


# It is my thesis that all the numbers in the square provide important information for comparing any pair of observers. Here are the input numbers for observers B and C:

# In[5]:

Bq1=qtd.Qq([2470000000,0,0.8869,0,1.8700,0,0,0])
Bq2=qtd.Qq([2580000000,0,3.9481,0,0,0.1064,0,0])
Bq1s=Bq1.product(q_scale)
Bq2s=Bq2.product(q_scale)
Bdq=Bq2s.dif(Bq1s).reduce()
Cq1=qtd.Qq([3470000000,0,1.1421,0,1.4220,0,1.3256,0])
Cq2=qtd.Qq([3580000000,0,4.2966,0,0,0.3643,1.3256,0])
Cq1s=Cq1.product(q_scale)
Cq2s=Cq2.product(q_scale)
Cdq=Cq2s.dif(Cq1s).reduce()
print(Bq1s)
print(Bq2s)
print(Bdq)
print(Cq1s)
print(Cq2s)
print(Cdq)


# No set of input numbers for two observers are **ever the same**. Two observers must be located in either a different place in time or a different place in space or both.

# In[6]:

Bdq2=Bq1s.dif(Bq2s).reduce().square()
Cdq2=Cq1s.dif(Cq2s).reduce().square()
print(Adq2)
print(Bdq2)
print(Cdq2)


# We are comparing apples to apples since the qtype, "QxS-QxS.reduce.sq", are the same. The first of the 8 terms are exactly the same, the I<sub>0</sub>. The reason is the delta time values were exactly the same. The first and third I<sub>2</sub> are exactly the same because their delta values were identical even though they had different z values. A different physical measurement was made for Observer B. The match is pretty good:

# In[7]:

(64.96 - 64.30)/64.60


# The error is about a percent. So while I reported 4 significant digits, only the first two can be trusted.

# The next experiment involved rotating the graph paper for Observer B. This should not change much other than the numbers that get plugged into the inteval calculation.

# ![](images/Billiards/shot_12_no_felt_rotated.png)

# In[8]:

BRotq1=qtd.Qq([2470000000,0,0.519,0,1.9440,0,0,0])
BRotq2=qtd.Qq([2580000000,0,3.9114,0,0.5492,0,0,0])
BRotdq2=BRotq1.product(q_scale).dif(BRotq2.product(q_scale)).reduce().square()
print(BRotdq2)
print(Bdq2)


# No surprise here: the graph paper will make a difference in the numbers used, but the distance is the same up to the errors made in the measuring process.

# ## Representations of Numbers  Versus Coordinate Transformation of Vectors

# This notebook is focused on space-time numbers that can be added, subtracted, multiplied, and divided. Formally, they are rank 0 tensors. Yet because space-time numbers have four slots to fill, it is quite easy to mistakenly view them as a four dimensional vector space over the mathematical field of real numbers with four basis vectors. Different representations of numbers changes the values of the numbers that get used, but not their meaning. Let's see this in action for a cylindrical representation of a number. Instead of $x$ and $y$, one uses $R \cos(\alpha)$ and $R \sin(\alpha)$, no change for $z$.

# ![](images/Billiards/shot_12_no_felt_polar.png)

# What needs to be done with the measurements done in cylindrical coordinates is to convert them to Cartesian, the proceed with the same calculations.

# In[9]:

import math

def cyl_2_cart(q1):
    """Convert a measurment made with cylindrical coordinates in angles to Cartesian cooridantes."""
    
    t = q1.dt.p - q1.dt.n
    r = q1.dx.p - q1.dx.n
    a = q1.dy.p - q1.dy.n
    h = q1.dz.p - q1.dz.n
    
    x = r * math.cos(a * math.pi / 180)
    y = r * math.sin(a * math.pi / 180)
    
    return qtd.Qq([t, x, y, h])


# For polar coordinates, measure directly the distance between the origin and the billiard ball. Then determine an angle. This constitutes a different approach to making a measurement.

# In[10]:

BPolarq1=cyl_2_cart(qtd.Qq([2470000000,0,2.0215,0, 68.0,0,0,0]))
BPolarq2=cyl_2_cart(qtd.Qq([2580000000,0,3.9414,0,1.2,0,0,0]))
BPolardq2=BPolarq1.product(q_scale).dif(BPolarq2.product(q_scale)).reduce().square()
print(BPolardq2)
print(Bdq2)


# Yet the result for the interval is the same: the positive time squared term is exactly the same since those numbers were not changed, and the negative numbers for the space terms were only different to the error in measurement.

# ## Observer B Boosted

# Give Observer B a Lorenz boost. All that is needed is to relocate Observer B in the second frame like so:

# ![](images/Billiards/shot_12_no_felt_boosted.png)

# To make the math simpler, presume all the motion is along $x$, not the slightest wiggle along $y$ or $z$. Constant motion between the frames shown is also presumed.

# What velocity is involved? THat would be the change in space, 2, over the time, a big number

# In[11]:

vx = 2/Bdq.dt.p
print(vx)


# This feels about right. The speed of observer B is about what a cube ball is.
# 
# Boost the delta by this velocity.

# In[17]:

Bdq_boosted = Bdq.boost(beta_x = vx)
print(Bdq_boosted)
print(Bdq_boosted.reduce())
print(Bdq)
print(Bdq_boosted.dif(Bdq).reduce())


# The last line indicates there is no difference between the boosted values of $y$ and $z$, as expected. Both the change in time and in space are negative. Moving in unison is a quality of simple boosts. The change in time is tiny. The change in space is almost 4, but not quite due to the work of the $\gamma$ factor that altered the time measurement.

# Compare the squares of the boosted with the non-boosted Observer B

# In[18]:

print(Bdq_boosted.square())
print(Bdq.square())


# Time and space are mixing together for the boosted frame. There are two huge numbers for $I_0$ and $I_2$ instead of a big number and about 65. Are they the same? Compare the reduced squares:

# In[19]:

print(Bdq_boosted.square().reduce())
print(Bdq.square().reduce())


# The reduced intervals are the same. The space-times-time terms are not. The difference between the space-times-time terms can be used to determine how Observer B boosted in moving relative to Observer B (calculation not done here). Even with out going into detail, the motion is only along x because that is the only term that changes.

# ## Observer C in a Gravity Field

# We know from the video of the billiard balls that they are in a gravity field since the eight ball drops into the pocket.
