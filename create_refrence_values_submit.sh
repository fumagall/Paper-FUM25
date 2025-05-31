cbsub -n 64 -J refrence_values -q BatchXL python create_reference_values_range.py <<< ""

cbsub -n 64 -J refrence_values_dt_02 -q BatchXL python create_reference_values_range.py --use_02_model <<< ""
