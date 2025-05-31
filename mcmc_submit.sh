python -c "for i in (1, 0.01, 0.0001): print(i)" | cbsub -J mcmc_ref -n 24 -q BatchXL python mcmc_am_ref.py
python -c "for i in (0.1, 0.01, 0.001): print(i)" | cbsub -J mcmc_short -n 24 -q BatchXL python mcmc_am_short.py
python -c "for i in (0.1, 0.01, 0.001): print(i)" | cbsub -J mcmc_long -n 24 -q BatchXL python mcmc_am_long.py 

python -c "for i in (0.1, 0.01, 0.001): print(i)" | cbsub -J mcmc_short_dt_02 -n 24 -q BatchXL python mcmc_am_short.py --use_02_model
python -c "for i in (0.1, 0.01, 0.001): print(i)" | cbsub -J mcmc_long_dt_02 -n 24 -q BatchXL python mcmc_am_long.py --use_02_model