# ipq - iPython Notebooks for Quaternion Math

sweetser@alum.mit.edu

These notebooks may help people play with both quaternions and space-time
number. The space-time numbers are my own invention, a variation on quaternions
that I study because it starts from group theory at its foundation. 

## For Newbies...

If you are new to the world of github and iPython, go watch a tutorial or two.
For the python, I recommend installing [anaconda from Continuum Analytics.](https://www.continuum.io/downloads)
It should install on Windows, Mac OS, and Linux machines. Once install, an
iPython jupyter notebook should "just work".

To get my software, open up a terminal and type:
```
> git clone https://github.com/dougsweetser/ipq
> cd ipq/q_notebooks
> jupyter notebook
```
This should start the notebook server in your default browser.

## The iPython Notebooks

The first job is to write the basic tools. That work can be found in the 
following notebook:

[Developing Quaternion and Space-time Number Tools for iPython3
](https://github.com/dougsweetser/ipq/blob/master/q_notebook/Q_tool_devo.ipynb)

One of the more interesting notational developments came from the realization
that all quaternions look identical. How could one know not to just add a
position quaternion to one for velocity? The software creates breadcrumbs of
the operations used to generate it. If it becomes too cumbersome, it may be
possible to shorten.

As an exercise, the space-time numbers are rotated and boosted in the following
notebook:

[Simple
Rotations](https://github.com/dougsweetser/ipq/blob/master/q_notebook/Rotations_of_Events_in_Space-time.ipynb)

Rotations in space are what quaternions are known for. There are a few places
in the published literature that claim real-valued quaternions cannot be used
to represent the Lorentz group. That claim may have been due to people assuming
the form must be the same as spatial rotation (U q U<sup>\*</sup>). A compact
Lie group is simpler than a non-compact one, and the form of the operation had
to change. It is not that bad, but it is different to do a boost. The details
of how to do a boost are in the "Tools" notebook.

*Observing Billiards Using Space-time Numbers*

This iPython notebook shows that space-time numbers can be used for simple
observerations of events.

[Billiards](https://github.com/dougsweetser/ipq/blob/master/q_notebook/billiard_calculations.ipynb)

The space-times-time equivalence class for gravity proposal is demonstrated.

Another iPython notebook for the gravity proposal is under development. It is
all about equivalence classes, ways we can say things are similar or exactly
the same.

[A Padantic Introduction to the Quaternion Gravity Proposal](https://github.com/dougsweetser/ipq/blob/master/q_notebook/QG_intro.ipynb)

This one just needs more time to flesh it out, but a lot of things are going on
for me now.
