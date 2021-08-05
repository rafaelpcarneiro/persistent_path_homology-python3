# Program to calculate the Persistent Path Homology of a network

Based on the paper [Persistent Path Homology of Directed Networks](https://arxiv.org/abs/1701.00565), from
the authors Samir Chowdhury and Facundo Mémoli, I will implement
in python their algorithm responsible to calculate the Persistent Path Homology from a network N=(X,A).

This technique os relevant if one is interested in a directed
graph which the weights of each edge of the graph are not
symmetrical.

## IMPORTANT
Here the field acting over the vector space is Z/2Z.

## HOW TO USE THIS SCRIPT

Given a network *N = (X, A)*, where *X* represents a
set of points (a subset of R^n) satisfying
- *X* is a *numpy array** with shape:
- X.shape[0] ---> size of our data
- X.shape[1] ---> dimension of the R^n space

and *A* a matrix of the weights such that
- *A* is a *numpy array** with shape:
- A.shape == (X.shape[0], X.shape[0]),

and A must satisfy the conditions:
- A[x,y] >= 0 for all x,y in {0, 1, 2, 3, ..., X.shape[0] - 1};
- A[x,y] == 0 iff x = y, for all x,y in {0, 1, 2, 3, ..., X.shape[0] - 1}.


Then, if someone wants to calculate the persistent path homology of
dimensions *0, 1, 2, ..., p* (with p >= 0), the following has 
to be done in a python interpreter (or in a script, it is up to you)

``` python
from persistent_path_homology import *
pph_X = PPH(X, A)
pph_X.ComputePPH()
print(pph.Pers)
```


Above, **pph_X** is an object of the class **PPH** and the method
**ComputePPPH()** calculates the persistent path homology,
which will be stored as one  atribute of **pph_X**,
namely **pph_X.Pers**. This atribute **pph_X.Pers** will be a
list containing all persistent path intervals. Each element of
this list will look like:
* pph_X.Pers[0] is a list containing persistent path features of dimension 0;
* pph_X.Pers[1] is a list containing persistent path features of dimension 1;
* pph_X.Pers[2] is a list containing persistent path features of dimension 2;
* ...

In case you wish to see the algorithm running step by step, you
must call the method **ComputePPH_printing_step_by_step()**. For
instance:

``` python
from persistent_path_homology import *
pph_X = PPH(X, A)
pph_X.ComputePPH_printing_step_by_step()
```


## TECHNICAL INFORMATION.
------------------------
1) The algorithm implemented here can be found
at:
> [Persistent Path Homology of Directed Networks](https://epubs.siam.org/doi/10.1137/1.9781611975031.75)
by the authors: Samir Chowdhury and Facundo Mémoli. (**Obs**: I just recently noticed that the authors
have their implementation of their algorithm here at Github. Their repository can be found here:
[authors code](https://github.com/samirchowdhury/pypph))

2) The algorithm proposed above do not
work properly if we do not set the regular paths of
dimension 0 to be marked. That is, it is mandatory to
mark this regular paths otherwise the persistent features
of dimension 0 won't be detected by the algorithm and also
it will lead to incorrect persistent diagrams.
This can be noticed by checking the algorithm proposed
at the paper
> [Computing Persistent Homology](https://geometry.stanford.edu/papers/zc-cph-05/) by the authors Afra Zomorodian and Gunnar Carlsson.

3) The software used to write this python script
was the magnificent **Doom-Emacs**, which is an
Emacs embedded with all functionalities of
VIM. I really recommend using emacs with
this framework. Down bellow I leave its github
repository:
> [Doom-Emacs](https://github.com/hlissner/doom-emacs) An Emacs framework for the stubborn martian hacker (as the author says =D)

4) This script has been tested using python3 on
a Ubuntu machine and it will make use of the library **numpy**:
> [Numpy](https://numpy.org/) Numerical computing tools for python

