
# coding: utf-8

# In[1]:




# In[5]:

from glob import glob
image_files = glob("images/eq_classes/*png")


# In[3]:

help(glob)


# In[8]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import basename


# In[43]:

eq_images = {}

for image_file in image_files:
    file_name = basename(image_file)
    eq_class_name = (file_name.split(sep='.'))[0]
    eq_images[eq_class_name] = mpimg.imread(image_file)
    
pprint(eq_images.keys())


# In[57]:

fig = plt.figure()
plt.rcParams["figure.figsize"] = [50, 20]

ax1 = fig.add_subplot(2, 5, 1)
ax1.imshow(eq_images['time_future'])
plt.axis('off')

ax2 = fig.add_subplot(2, 5, 2)
ax2.imshow(eq_images['space_right'])
plt.axis('off');

ax3 = fig.add_subplot(2, 5, 3)
ax3.imshow(eq_images['causality_time-like'])
plt.axis('off');

ax4 = fig.add_subplot(2, 5, 4)
ax4.imshow(eq_images['space-times-time_past-right_exact'])
plt.axis('off');

ax5 = fig.add_subplot(2, 5, 6)
ax5.imshow(eq_images['norm_of_unity_less_than_unity'])
plt.axis('off');

ax10 = fig.add_subplot(2, 5, 5)
ax10.imshow(eq_images['norm_of_unity_less_than_unity'])
plt.axis('off');

ax6 = fig.add_subplot(2, 5, 7)
ax6.imshow(eq_images['norm_of_unity_less_than_unity'])
plt.axis('off');
ax7 = fig.add_subplot(2, 5, 8)
ax7.imshow(eq_images['norm_of_unity_less_than_unity'])
plt.axis('off');
ax8 = fig.add_subplot(2, 5, 9)
ax8.imshow(eq_images['norm_of_unity_less_than_unity'])
plt.axis('off');


# In[12]:

from pprint import pprint


# In[13]:

pprint(eq_images)


# In[31]:

help(str.split)


# In[34]:

f = "foo.bar"
g = (f.split(sep='.'))[0]
print(g)


# In[ ]:



