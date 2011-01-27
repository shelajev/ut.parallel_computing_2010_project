====================================
Pagerank Solution in Python with MPI
====================================

.. contents::

Running in Ubuntu
-----------------

This is how to run it in a clean Ubuntu environment.

Setting up Python and MPI
~~~~~~~~~~~~~~~~~~~~~~~~~

Commands should be run in superuser mode, this can be done via::
    
    sudo do_something
    
    su -c 'do_something'

Install python with development headers, if they're not installed already::
    
    sudo install python2.6 python2.6-dev

Install an implementation of MPI-2::
	
	sudo install mpich2

Make file ~/.mpd.conf with some secret word:: 
	
	echo "MPD_SECRETWORD=${RANDOM}z${RANDOM}z${RANDOM}" > $HOME/.mpd.conf
	chmod 600 $HOME/.mpd.conf

Install pip and setuptools, this helps to install python libraries easily::
		
    sudo apt-get install python-pip python-setuptools
        
Install mpi4py and necessary libraries::

    sudo apt-get install libssl-dev
	pip install mpi4py

Downloading source
~~~~~~~~~~~~~~~~~~

Install git and necessary libraries::

    sudo apt-get install git-core python-numpy python-scipy

Get the source::

    git clone git://github.com/shelajev/ut.parallel_computing_2010_project_newgoogle_py.git pagerank

Run it::

    cd pagerank
    mpirun -np 4 ./pagerank.py
    
    # if it complains about mpd not running, then run
    mpd --daemon


Tips for running on a cluster
-----------------------------

Things that you should keep in mind.

PATH
~~~~

The PATH should be the same on every node.
For example if you run this with mpirun on a cluster and your environment setup is 
done in a startup script - that script might not run. This means a different
version of python could be run or different versions of libraries are loaded.
There's a check for python version in pagerank.py, that's really not necessary
as this should run with newer versions of python as well, but it ensures that
the same version of python will be run on each node.

Depending on which mpirun there are ways to ensure proper PATH setup::

    # for MPICH2 mpirun
    mpirun -np 8 -envlist PATH ./pagerank.py
    
    # for openmpi mpirun, "-x" exports a single environment variable
    mpirun -np 8 -x PATH ./pagerank.py
    

Description of Calculator
-------------------------

Calculator is a general purpose same height matrix calculator running
on top of MPI. There are multiple nodes for doing the calculation and
one master node for sending the commands to the nodes.

The matrices are split rows wise. This means one node gets the top part
another the middle and the last the bottom part::

     _______                                  _______
    |       |                                [___A1__]
    |   A   |   --- Calculator.Set(A) --->   [___A2__]
    |_______|                                [___A3__]

Most simpler calculations can be done with partial matrices::

     _______      _______       _________
    [_A1____]    [_B1____]     [_A1_+_B1_]
    [_A2____] +- [_B2____]  =  [_A2_+_B2_]
    [_A3____]    [_B3____]     [_A3_+_B3_]


For multiplication the matrix B must be collected on one side::

     _______      _______                   ________
    [_A1____]    [_B1____]     [A1] * B    [_A1_*_B_]
    [_A2____] *  [_B2____]  =  [A2] * B  = [_A2_*_B_]
    [_A3____]    [_B3____]     [A3] * B    [_A3_*_B_]

There is an optimization for sending that content to the specific
node that it only needs (x- shows where A contains value )::

     _______      _______                             ________
    [_xx____]    [_B1____]     [A1] * ( B1 )      =  [_A1_*_B_]
    [___xx__] *  [_B2____]  =  [A2] * ( B2 )      =  [_A2_*_B_]
    [_x____x]    [_B3____]     [A3] * ( B1 & B3 ) =  [_A3_*_B_]

This saves us some communication.

Adding a new operation to Calculator
------------------------------------

Operating with partial matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic structure of the Calculator and
CalculatorNode-s is. There is one Calculator
instance and multiple CalculatorNode-s.

Calculators purpose is to pass commands to
CalculatorNodes and distribute and collect data.

Each CalculatorNode holds a partial matrix (row wise).
This means that each Node holds specific rows, this 
is determined in the Setup of Calculator.

First step for setting up a new operation is to
add a new tag in the beginning of calculator.py.
Let's add OP_MAGIC to complex commands.

The tg() function ensures that each command gets
an unique id.

Next step is to add appropriate function to Calculator.
We will add this to the end of the class::

    def Magic(self, r, a, s):
        """ Does some magic with a and s, stores the result in r """
        ras = self.ras(r, a, s)
        self.Do(ras, OP_MAGIC)

The convention for commands that generate a new matrix or result is to
give a new place to store the result. This means we get greater flexibility
how we can use the commands.

"self.ras(r,a,s)" command converts matrix names 'r', 'a' to ids. This is just a convenience
function. This also checks whether 'a' exists already. There are some similar commands: ras - result, matrix, scalar; rab - result, matrix, matrix; ra - result, matrix.

"self.Do()" sends the command to all Nodes.

Now we need to capture the command on the Nodes.

First we'll add redirection for this command in CalculatorNode.loop::

    ops = {
        ...
        OP_DOT  : self._dot,
        OP_MEX  : self._mex,
        # complex commands
        OP_PREPARE_PAGERANK : self._prepare_pagerank,
        OP_MAGIC : self._magic,
    }

This translates the tag into a function "_magic". Also add this function just before
the header for main loops::

    def _magic(self, data):
        r, a, b = data
        self.Magic(r, a, b)

This "_" prefixed functions unpack the input data and translate them to their respective values. Now we also need to add "Magic" function::

    def Magic(self, result, a, value):
        A = self.get(a)
        A = (A + value) * value
        self.set(r, A)

The "self.get(a)" gets the respective partial matrix from matrix list. 
Then we do some magic calculations with it and store the result with "self.set(r, A)".

Now we can execute the new computation.

Communicating with other nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's also add some internal node calculation here.

Add a new internal operation tag "_OP_MAGIC". 
Also let's modify Magic function::

    def Magic(self, result, a, value):
        A = self.get(a)
        A = A + value
        value = value + A[0,0]
        newvalue = self.comm.sendrecv( value, self.left,  _OP_MAGIC,
                                        None, self.right, _OP_MAGIC )
        A = A * newvalue
        self.set(r, A)
        
Nodes have been automatically setup on a circle and "self.left", "self.right" store
the appropriate neighbor node ids. 
We have to use sendrecv or non blocking calls as we don't want our program
to run into a deadlock.

