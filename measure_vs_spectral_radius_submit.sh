seq 0 4 |cbsub -n 48 -J uncoupled_spectral_radius -q BatchXL python measure_vs_spectral_radius.py --n_samples 1000 --use_uncoupled
seq 0 4 |cbsub -n 48 -J random_spectral_radius -q BatchXL python measure_vs_spectral_radius.py --n_samples 1000 

seq 0 4 | cbsub -n 48 -J uncoupled_spectral_radius_dt_02 -q BatchXL python measure_vs_spectral_radius.py --use_02_model --n_samples 1000
seq 0 4 | cbsub -n 48 -J random_spectral_radius_dt_02 -q BatchXL python measure_vs_spectral_radius.py --use_02_model --n_samples 1000 --use_uncoupled

seq 0 4 | cbsub -n 48 -J uncoupled_spectral_radius_dt_02 -q BatchXL python measure_vs_spectral_radius.py --n_nodes 10 --use_02_model --n_samples 1000 --use_uncoupled
seq 0 4 | cbsub -n 48 -J random_spectral_radius_dt_02 -q BatchXL python measure_vs_spectral_radius.py --n_nodes 10 --use_02_model --n_samples 1000

seq 0 4 | cbsub -n 48 -J uncoupled_spectral_radius_dt -q BatchXL python measure_vs_spectral_radius.py --n_nodes 10 --n_samples 1000 --use_uncoupled
seq 0 4 | cbsub -n 48 -J random_spectral_radius_dt -q BatchXL python measure_vs_spectral_radius.py --n_nodes 10 --n_samples 1000

seq 4 4 | cbsub -n 48 -J uncoupled_spectral_radius_single_dt_02_10 -q BatchXL python measure_vs_spectral_radius_single.py --n_nodes 10 --use_02_model --n_samples 100000 --spectral_radius 0.4 --use_uncoupled
seq 4 4 | cbsub -n 48 -J uncoupled_spectral_radius_single_dt_02 -q BatchXL python measure_vs_spectral_radius_single.py --n_nodes 20 --use_02_model --n_samples 100000 --spectral_radius 0.4 --use_uncoupled

seq 4 4 | cbsub -n 48 -J uncoupled_spectral_radius_single_dt_02_10 -q BatchXL python measure_vs_spectral_radius_single.py --n_nodes 10 --use_02_model --n_samples 100000 --spectral_radius 0.4

seq 4 4 | cbsub -n 48 -J uncoupled_spectral_radius_single_dt_10 -q BatchXL python measure_vs_spectral_radius_single.py --n_nodes 10 --n_samples 50000 --spectral_radius 0.4 --use_uncoupled
seq 4 4 | cbsub -n 48 -J uncoupled_spectral_radius_single_dt -q BatchXL python measure_vs_spectral_radius_single.py --n_nodes 20 --n_samples 50000 --spectral_radius 0.4 --use_uncoupled