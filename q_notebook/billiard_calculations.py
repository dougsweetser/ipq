
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

import Q_tool_devo as qtd;
Aq1=qtd.Qq([1470000000,0,1.1421,0,1.4220,0,0,0])
Aq2=qtd.Qq([1580000000,0,4.2966,0,0,0.3643,0,0])


# In[2]:

q_scale = qtd.Qq([2.2119,0,0,0,0,0,0,0], qtype="S")
Aq1s=Aq1.product(q_scale)
Aq2s=Aq2.product(q_scale)
print(Aq1s)


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
Cq1=qtd.Qq([3470000000,0,1.1421,0,1.4220,0,1.3256,0])
Cq2=qtd.Qq([3580000000,0,4.2966,0,0,0.3643,1.3256,0])


# No set of input numbers for two observers are **ever the same**. Two observers must be located in either a different place in time or a different place in space or both.

# In[6]:

Bdq2=Bq1.product(q_scale).dif(Bq2.product(q_scale)).reduce().square()
Cdq2=Cq1.product(q_scale).dif(Cq2.product(q_scale)).reduce().square()
print(Adq2)
print(Bdq2)
print(Cdq2)


# We are comparing apples to apples since the qtype, "QxS-QxS.reduce.sq", are the same. The first of the 8 terms are exactly the same, the I<sub>0</sub>. The reason is the delta time values were exactly the same. The first and third I<sub>2</sub> are exactly the same because their delta values were identical even though they had different z values. A different physical measurement was made for Observer B. The match is pretty good:

# In[14]:

((64.96 - 64.30)/64.60)**0.5


# The error is about a tenth of a percent. So while I reported 4 significant digits, only the first three can be trusted.

# The next experiment involved rotating the graph paper for observer B. This should not change much other than the numbers that get plugged into the inteval calculation.

# ![](images/Billiards/shot_12_no_felt_rotated.png)

# In[8]:

print(qtd.EQ(Adq2,Bdq2))


# In[16]:

BRotq1=qtd.Qq([2470000000,0,0.519,0,1.9440,0,0,0])
BRotq2=qtd.Qq([2580000000,0,3.9114,0,0.5492,0,0,0])
BRotdq2=BRotq1.product(q_scale).dif(BRotq2.product(q_scale)).reduce().square()
print(BRotdq2)
print(Bdq2)


# ![](images/Billiards/shot_12_no_felt_polar.png)

# In[ ]:



