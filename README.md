# Restricted Boltzmann Machine for co-reference resolution

The task of the current project is co-reference resolution using representation learning. In other words, to learn to co-refer mentions in a speech context, e.g. phrases, dialogs. 

The project is consisted of two phases: 
1. Finding the mentions such as _Jack_, _the book_, _Marie_, _he_ and _it_ in the phrase “Jack gave the book back to Marie. He had borrowed it last week”.
2. Modeling the co-reference representation and training a model for further resolution. 

In this project we concentrate on the second phase. 

### Representation Learning

Representation Learning, sometimes called Feature Learning, is a set of techniques that learn a transformation of raw data input to a representation that can be effectively exploited in machine learning tasks. Imagine a database of millions of phrases in which the potential mentions (references are traditionally called mention) are extracted and each instance’s features are gathered. 

```Example: Beatrice saw Mr. Orwell yesterday. He is her professor at university.```

In this example, “He” and “her” refer to Mr. Orwell and Beatrice in order.

### Restricted Boltzmann Machine

A Restricted Boltzmann Machine (RBM) is a stochastic artificial neural network that can learn a probability distribution over its set of inputs. An RBM consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections (as its major graphical specifications). More information may be found on the Web.

### One challenge

One of my challenges in implementing Restricted Boltzmann Machine for the provided data set was to convert the binary RBM, which had various implementations already available, into a Gaussian one. This way, we could save more time and work more on the experiments and the optimization challenges. 

There are (up to 2015) quite different solutions to convert a binary RBM into a Gaussian RBM in order to be able to train the network over real-valued datasets. [One way](http://www.cs.toronto.edu/~hinton/csc2535/notes/lec4new.pdf) to model an integer-valued variable is to make N identical copies of a binary unit. [Another way](https://blog.safaribooksonline.com/2014/02/10/pylearn2-regression-3rd-party-data/) suggests using a Multilayer Perceptron (MLP) to have the final layer to be one that models a continuous layer. Finally, [another solution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.478.6892&rep=rep1&type=pdf) proposes a few improvements to the conventional training methods for GBRBM.


### Requirements 
  * [Python 2.7](https://www.python.org/download/releases/2.7/)
  * [Theano](http://deeplearning.net/software/theano/)
  * [Octave](https://www.gnu.org/software/octave/)
  * [pylearn2](http://deeplearning.net/software/pylearn2/)


### Confidentiality 
This repository only covers a small part of the project, mostly general calculations. It is hardly reusable for the same task.
