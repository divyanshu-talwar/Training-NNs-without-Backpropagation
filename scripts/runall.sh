#! /usr/bin/bash

# Classification Datasets Backprop Implementation Run
python backprop_implementation.py --is_classification True --dataset iris --log_file iris_backprop.log --plot_file iris_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset digits --log_file digits_backprop.log --plot_file digits_backprop.png --hidden_units 36
python backprop_implementation.py --is_classification True --dataset wine --log_file wine_backprop.log --plot_file wine_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset breast_cancer --log_file breast_cancer_backprop.log --plot_file breast_cancer_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset tic-tac-toe --log_file tic-tac-toe_backprop.log --plot_file tic-tac-toe_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset soybean-small --log_file soybean-small_backprop.log --plot_file soybean-small_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset shuttle --log_file shuttle_backprop.log --plot_file shuttle_backprop.png --hidden_units 4
python backprop_implementation.py --is_classification True --dataset glass --log_file glass_backprop.log --plot_file glass_backprop.png --hidden_units 8
python backprop_implementation.py --is_classification True --dataset seeds --log_file seeds_backprop.log --plot_file seeds_backprop.png --hidden_units 6
python backprop_implementation.py --is_classification True --dataset fertility --log_file fertility_backprop.log --plot_file fertility_backprop.png --hidden_units 5

# Regression Datasets Backprop Implementation Run
python backprop_implementation.py --dataset boston --log_file boston_backprop.log --plot_file boston_backprop.png --hidden_units 100
python backprop_implementation.py --dataset linnerud --log_file linnerud_backprop.log --plot_file linnerud_backprop.png --hidden_units 2
python backprop_implementation.py --dataset real-estate-valuation --log_file real-estate-valuation_backprop.log --plot_file real-estate-valuation_backprop.png --hidden_units 12
python backprop_implementation.py --dataset energy-efficiency --log_file energy-efficiency_backprop.log --plot_file energy-efficiency_backprop.png --hidden_units 6
python backprop_implementation.py --dataset stock-portfolio-performance --log_file stock-portfolio-performance_backprop.log --plot_file stock-portfolio-performance_backprop.png --hidden_units 9
python backprop_implementation.py --dataset concrete-slump --log_file concrete-slump_backprop.log --plot_file concrete-slump_backprop.png --hidden_units 4
python backprop_implementation.py --dataset daily-demand-forecasting --log_file daily-demand-forecasting_backprop.log --plot_file daily-demand-forecasting_backprop.png --hidden_units 91
python backprop_implementation.py --dataset concrete-compressive-strength --log_file concrete-compressive-strength_backprop.log --plot_file concrete-compressive-strength_backprop.png --hidden_units 6
python backprop_implementation.py --dataset airfoil-self-noise --log_file airfoil-self-noise_backprop.log --plot_file airfoil-self-noise_backprop.png --hidden_units 7
python backprop_implementation.py --dataset o-ring --log_file o-ring_backprop.log --plot_file o-ring_backprop.png --hidden_units 6

# Classification Datasets ADMM Implementation Run
python admm_implementation.py --is_classification True --dataset iris --log_file iris_admm.log --plot_file iris_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset digits --log_file digits_admm.log --plot_file digits_admm.png --hidden_units 36
python admm_implementation.py --is_classification True --dataset wine --log_file wine_admm.log --plot_file wine_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset breast_cancer --log_file breast_cancer_admm.log --plot_file breast_cancer_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset tic-tac-toe --log_file tic-tac-toe_admm.log --plot_file tic-tac-toe_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset soybean-small --log_file soybean-small_admm.log --plot_file soybean-small_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset shuttle --log_file shuttle_admm.log --plot_file shuttle_admm.png --hidden_units 4
python admm_implementation.py --is_classification True --dataset glass --log_file glass_admm.log --plot_file glass_admm.png --hidden_units 8
python admm_implementation.py --is_classification True --dataset seeds --log_file seeds_admm.log --plot_file seeds_admm.png --hidden_units 6
python admm_implementation.py --is_classification True --dataset fertility --log_file fertility_admm.log --plot_file fertility_admm.png --hidden_units 5

# Regression Datasets ADMM Implementation Run
python admm_implementation.py --dataset boston --log_file boston_admm.log --plot_file boston_admm.png --hidden_units 100
python admm_implementation.py --dataset linnerud --log_file linnerud_admm.log --plot_file linnerud_admm.png --hidden_units 2
python admm_implementation.py --dataset real-estate-valuation --log_file real-estate-valuation_admm.log --plot_file real-estate-valuation_admm.png --hidden_units 12
python admm_implementation.py --dataset energy-efficiency --log_file energy-efficiency_admm.log --plot_file energy-efficiency_admm.png --hidden_units 6
python admm_implementation.py --dataset stock-portfolio-performance --log_file stock-portfolio-performance_admm.log --plot_file stock-portfolio-performance_admm.png --hidden_units 9
python admm_implementation.py --dataset concrete-slump --log_file concrete-slump_admm.log --plot_file concrete-slump_admm.png --hidden_units 4
python admm_implementation.py --dataset daily-demand-forecasting --log_file daily-demand-forecasting_admm.log --plot_file daily-demand-forecasting_admm.png --hidden_units 91
python admm_implementation.py --dataset concrete-compressive-strength --log_file concrete-compressive-strength_admm.log --plot_file concrete-compressive-strength_admm.png --hidden_units 6
python admm_implementation.py --dataset airfoil-self-noise --log_file airfoil-self-noise_admm.log --plot_file airfoil-self-noise_admm.png --hidden_units 7
python admm_implementation.py --dataset o-ring --log_file o-ring_admm.log --plot_file o-ring_admm.png --hidden_units 6