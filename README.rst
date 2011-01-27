=============================
Pagerank Solution in Python
=============================
Project in Parallel Computing
-----------------------------

Running in Ubuntu
=================

Setting up python and mpi
-------------------------

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
------------------

Install git and necessary libraries::

    sudo apt-get install git-core python-numpy python-scipy

Get the source::

    git clone git://github.com/shelajev/ut.parallel_computing_2010_project_newgoogle_py.git pagerank

Run it::

    cd pagerank
    mpirun -np 4 ./pagerank.py
    
    # if it complains about mpd not running, then run
    mpd --daemon
    
