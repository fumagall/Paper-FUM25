seq 0 4 | cbsub -J ci_long -n 48 -q BatchXL python create_correlation_integral_long.py
seq 0 4 | cbsub -J ci_short -n 48 -q BatchXL python create_correlation_integral_short.py
seq 0 4 | cbsub -J ci_ref -n 48 -q BatchXL python create_correlation_integral_ref.py

seq 0 4 | cbsub -J ci_long_dt_02 -n 48 -q BatchXL python create_correlation_integral_long.py --use_02_model
seq 0 4 | cbsub -J ci_short_dt_02 -n 48 -q BatchXL python create_correlation_integral_short.py --use_02_model