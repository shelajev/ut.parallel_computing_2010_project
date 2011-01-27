====================================
Pagerank Solution in Python with MPI
====================================


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


Running in Ubuntu
-----------------

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
	chmod 600 $HOME/.mpd.confpython-numpy python-scipy

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
    
