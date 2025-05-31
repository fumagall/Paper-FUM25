# Naming scheme
Most files come with 3 file extensions
* file_name.py (python)
* file_name_submit.sh (bash)
* file_name_figure_x.ipynb (jupyter)

The bash script just calls the python file to submit it on a cluster using cbsub (LSF).

After a sucesful run, the jupyter file can be executed to show too corrisponding figure.

Be aware that files depend on each other, if you do not want to use to presimulated data (see data section below)

# Create Figures without Data

For all figures except 6b, you can just run the .ipynb file in jupyter lab with the paper name.

For the figure 6b, the data was too large to up load, and thus, it must be simulated first. For this, run create_figure_6b_submit.sh to run the script via cbsub (LSF) on a cluster or to run it locally execute create_figure_6b.py file directly with the arguments specified in the create_figure_6b_submit.sh file. Afterwards, the .ipynb file can be run.

# Data creation

The script do depend on each other, and thus, execuion order is important.

1. run create_correlation_integral_submit.sh
2. run create_histogram_submit.sh
3. run create_refrence_values_submit.sh
4. run mcmc_submit.sh
5. run shrink_and_inflate_submit.sh
6. run sweep_theta_1d_submit.sh
7. run measure_vs_spectral_radius_submit.sh

Finally, you reproduced all the files in data and results! :)