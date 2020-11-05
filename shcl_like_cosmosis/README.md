Run:
cosmosis params.ini

to get a single likelihood with results in the directory "output",
or, e.g. :

mpirun -n 2 cosmosis --mpi params.ini -p runtime.sampler=star

to get a slice in each parameter in "chain.txt", and then run:
postprocess chain.txt -o plots
to make plots in the directory "plots".  Note that the flat IA eta plot
is expected since the fiducial IA A parameter is zero.