# code of causal graph discovery

## requirement
python==3.6
pytorch==1.0.1
scikit-learn==0.24.0

## key_directory                   
- /region_features_NYC.csv #regional attributes data of  NYC 
- /region_features_Chi.csv #regional attributes data of  Chi
- /region_features_Sea.csv #regional attributes data of  Sea
- /castle/ # causal discovery package
- /causal discovery.py              # code for causal graph discovery


# code of CEVAE-based feature recovery model

## requirement
python==3.6
pytorch==1.0.1
scikit-learn==0.24.0

## key_directory
- /region_features_NYC.csv #regional attributes data of  NYC 
- /region_features_Chi.csv #regional attributes data of  Chi
- /region_features_Sea.csv #regional attributes data of  Sea
- /best_graph_DAWG_NYC.txt #causal graph of NYC
- /best_graph_prun_NYC.txt #pruned causal graph of NYC
- /CEVAE.py #code of CEVAE
- /VAE.py #code of VAE
- /AE.py #code of AE
  
# code of OD prediction

## requirement
python==3.6
pytorch==1.0.1
dgl==0.4.1
scikit-learn==0.24.0


## key_directory
- ./configure        # configure file for experiments
- ./data             # all related data
	- /NYC
	- /Chicago
	- /Seattle
- ./src              # all code
	- /dataloader
		- /dataloader.py 
	- /transfer.py


## introduction for configure
- experiment      # name of running experiment
- src_city        # source city
- tar_city        # target city
- input           # what kind of input for GNN: embdding, complete, incomplete
- run_mode        # train or test
- model           # init or load
- model_directory # for saving or loading model
- experiment_directory    # for saving exp results
- train_data_fraction     # how many OD could use to fine tune model and knowledges stillation
- valid_data_fraction     # for checking overfitting
- num_layers              # GNN hyper-parameters
- num_hidden
- num_out
- heads
- feat_drop
- attn_drop
- negative_slope
- residual
- batch_size
- activation
- epochs          # number of training epochs
- learning_rate   # learning_rate for source city OD prediction model
- flearning_rate  # learning_rate for target city fine tuning
- optimizer       # Adam OR SDG etc.
- random_seed
- random_seed_data
