import sys 
import os 
sys.path.append(os.getcwd())

import numpy as np 
import json 
from run import run_pysustain, eval_pysubebm
import yaml
import warnings

warnings.filterwarnings("ignore")

def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Get directories correct
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current working directory: {base_dir}")
    data_dir = os.path.join(base_dir, "data")

    # Read parameters from command line arguments
    filename = sys.argv[1]
    print(f"Processing with {filename}")
    data_file = os.path.join(data_dir, f"{filename}.csv")
    if not os.path.isfile(data_file):
        print(f"Error: Data file {data_file} does not exist.")
        sys.exit(1)

    # Parameters
    config = load_config()

    # Number of independent optimization attempts in greedy ascent
    N_MCMC=config['N_MCMC']
    N_SHUFFLE=config['N_SHUFFLE']
    BURN_IN=config['BURN_IN']
    THINNING=config['THINNING']
    OUTPUT_DIR=config['OUTPUT_DIR']
    MCMC_SEED = config['MCMC_SEED']
    N_SUBTYPE_SHUFFLE = config['N_SUBTYPE_SHUFFLE']
    MAX_N_SUBTYPES = config['N_SUB_HI']
    N_FOLDS = config['SUSTAIN_EVAL_N_FOLD']
    PRIOR_N = config['PRIOR_N']
    PRIOR_V = config['PRIOR_V']
    N_MCMC_SUSTAIN = config['N_MCMC_SUSTAIN']
    Z_SCORE_NORM = config['Z_SCORE_NORM']
    rng = np.random.default_rng(MCMC_SEED)

    # Get true order and true stages dict
    with open(os.path.join(base_dir, "true_order_and_stages.json"), "r") as f:
        true_order_and_stages = json.load(f)
    metadata = true_order_and_stages[filename]
    n_subtypes = metadata['N_SUB']
    true_order_matrix = metadata['TRUE_ORDERINGS']
    true_subtype_assignments = metadata['TRUE_SUBTYPE_ASSIGNMENTS']
    true_stage_assignments = metadata['TRUE_STAGE_ASSIGNMENTS']

    if 'Esn_kjOrdinalDM_xnjNormal' in filename:
        estimate_n_subtypes = True
    else:
        estimate_n_subtypes = False

    ###################################################################################
    # Step1: BEBMS
    ###################################################################################
    random_state = rng.integers(0, 2**32 - 1)
    # with labels:
    for with_labels in [True, False]:
        # for with_labels in [True]:
        if with_labels:
            output_dir = os.path.join(OUTPUT_DIR, 'pysubebm_with_labels')
        else:
            output_dir = os.path.join(OUTPUT_DIR, 'pysubebm')
        eval_pysubebm(
            filename = filename,
            data_file= data_file,
            n_subtypes=n_subtypes,
            true_order_matrix=true_order_matrix,
            true_subtype_assignments=true_subtype_assignments,
            output_dir=output_dir,
            # n_iter=1000,
            n_iter=N_MCMC,
            n_shuffle=N_SHUFFLE,
            n_subtype_shuffle=N_SUBTYPE_SHUFFLE,
            max_n_subtypes=MAX_N_SUBTYPES,
            burn_in=BURN_IN,
            thinning=THINNING,
            seed=random_state,
            prior_n=PRIOR_N,
            prior_v=PRIOR_V,
            N_FOLDS=N_FOLDS,
            save_results=False, # no worries, we will save inside the eval_pysubebm
            with_labels=with_labels,
            estimate_n_subtypes=estimate_n_subtypes,
            z_score_norm=Z_SCORE_NORM
        )

    ###################################################################################
    # Step2: PYSUSTAIN
    ###################################################################################
    random_state = rng.integers(0, 2**32 - 1)
    run_pysustain(
        filename=filename,
        data_file=data_file, # full path to data 
        sustainType='mixture_GMM', # 'mixture_GMM' 'mixture_KDE'
        n_startpoints=config['N_STARTPOINTS'],
        output_dir=os.path.join(OUTPUT_DIR, 'sustain_gmm'),
        true_n_subtypes=n_subtypes,
        max_n_subtypes=MAX_N_SUBTYPES,
        # n_iter=1000,
        n_iter=N_MCMC_SUSTAIN,
        random_state=random_state,
        SUSTAIN_EVAL_N_FOLD = N_FOLDS,
        true_order_matrix=true_order_matrix,
        true_subtype_assignments=true_subtype_assignments,
        estimate_n_subtypes=estimate_n_subtypes,
    )

    random_state = rng.integers(0, 2**32 - 1)
    run_pysustain(
        filename=filename,
        data_file=data_file, # full path to data 
        sustainType='mixture_KDE', # 'mixture_GMM' 'mixture_KDE'
        n_startpoints=config['N_STARTPOINTS'],
        output_dir=os.path.join(OUTPUT_DIR, 'sustain_kde'),
        true_n_subtypes=n_subtypes,
        # n_iter=1000,
        n_iter=N_MCMC_SUSTAIN,
        max_n_subtypes=MAX_N_SUBTYPES,
        random_state=random_state,
        SUSTAIN_EVAL_N_FOLD = N_FOLDS,
        true_order_matrix=true_order_matrix,
        true_subtype_assignments=true_subtype_assignments,
        estimate_n_subtypes=estimate_n_subtypes,
    )
