import pandas as pd 
import numpy as np 
import os 
import json 
from pySuStaIn.MixtureSustain import MixtureSustain
from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
import bebms.utils as utils
import time
import warnings
import pickle 
from bebms import run_bebms, cross_validatation
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict
from scipy.optimize import linear_sum_assignment
import copy

def get_tau(
        true_order_matrix: np.ndarray, 
        best_order_matrix: np.ndarray,
) -> Tuple[float]:
    n = len(best_order_matrix)
    dist = np.zeros((n, n))
    # i can safely use the sequence results because they are the indices of the fixed input biomarker array!
    for i in range(n):
        for j in range(n):
            dist[i,j]= utils.normalized_kendalls_tau_distance(
                best_order_matrix[i], true_order_matrix[j]) 
    # This finds the best matching: estimated_indices[i] -> true_indices[i]
    estimated_indices, true_indices = linear_sum_assignment(dist)
    # Calculate the matched Kendall's Tau
    tau = dist[estimated_indices, true_indices].mean()
    return tau

warnings.filterwarnings("ignore")

def run_pysustain(
        filename:str,
        data_file:str, # full path to data 
        sustainType:str, # 'mixture_GMM' 'mixture_KDE'
        n_startpoints:int,
        output_dir:str,
        true_n_subtypes:int,
        max_n_subtypes:int,
        n_iter:int,
        random_state:int,
        SUSTAIN_EVAL_N_FOLD,
        true_order_matrix:np.ndarray = None,
        true_subtype_assignments:np.ndarray= None,
        estimate_n_subtypes:bool=False
) -> None:
    
    start_time = time.time()
    RESULTS_DIR = os.path.join(output_dir, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # read file
    df = pd.read_csv(data_file)
    df.drop(columns=['participant'], inplace=True)

    # extract data 
    biomarker_labels = list(df.columns)[:-1]
    data_matrix = df.to_numpy()
    data = data_matrix[:, :-1].astype(np.float64)
    target = data_matrix[:, -1].astype(np.int64)
    diseased_mask = (target == 1) # ad

    # prepare for sustain analysis 
    if sustainType == "mixture_GMM":
        mixtures = fit_all_gmm_models(data, target)
    elif sustainType == "mixture_KDE":
        mixtures = fit_all_kde_models(data, target)

    # Extract likelihoods for each biomarker
    L_yes = np.zeros(data.shape)
    L_no = np.zeros(data.shape)
    for i in range(data.shape[1]):
        if sustainType == "mixture_GMM":
            L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
        elif sustainType == "mixture_KDE":
            L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

    # -------------------------------------------------------------------------
    # Full MCMC run (20k mcmc iterations)
    # -------------------------------------------------------------------------
    # parameter setting 
    N_startpoints = n_startpoints  # Number of starting points for optimization
    N_S_max = true_n_subtypes  # Maximum number of subtypes (since you mentioned 2 orderings)
    N_iterations_MCMC = n_iter  # Number of MCMC iterations
    dataset_name = filename
    use_parallel_startpoints = True

    sustain_model = MixtureSustain(
        L_yes, 
        L_no, 
        biomarker_labels,  # biomarker labels
        N_startpoints, 
        N_S_max, 
        N_iterations_MCMC, 
        output_dir, 
        dataset_name, 
        use_parallel_startpoints
    )

    # the ml_subtype and ml_stage are based on the 1000 selective mcmc posterors, sort of 
    # sustain might be doing majority vote here? I am not sure, in fact. 
    samples_sequence, samples_f, ml_subtype, _, ml_stage, _, _ = sustain_model.run_sustain_algorithm(plot=False)
    pickle_filename = os.path.join(sustain_model.output_folder, 'pickle_files',  f'{sustain_model.dataset_name}_subtype{sustain_model.N_S_max-1}.pickle')

    with open(pickle_filename, 'rb') as f:
        loaded_variables = pickle.load(f)
    samples_likelihood = loaded_variables['samples_likelihood']
    best_mcmc_idx = np.argmax(samples_likelihood.flatten())

    # first should be the ml_sequence_em
    first_sample = samples_sequence[:, :, 0]
    ml_from_pickle = loaded_variables['ml_sequence_EM']

    # mcmc best
    best_sample = samples_sequence[:, :, best_mcmc_idx]

    # EM-only: force the assignment to use just the EM sample (index 0)
    ml_subtype_em, p_ml_subtype_em, ml_stage_em, p_ml_stage_em, *_ = \
        sustain_model.subtype_and_stage_individuals_newData(
            L_yes, L_no, samples_sequence, samples_f, N_samples=1
    )
    # Are any of these the same?
    print("First == Best?", np.array_equal(first_sample, best_sample))
    print("First == Pickle ML?", np.array_equal(first_sample, ml_from_pickle))
    print("Best == Pickle ML?", np.array_equal(best_sample, ml_from_pickle))

    best_order_matrix_em = copy.deepcopy(ml_from_pickle) # n_subtypes, n_biomarkers
    best_order_matrix_mcmc = copy.deepcopy(best_sample)
    ml_subtype_em = ml_subtype_em.flatten()
    ml_stage_em = ml_stage_em.flatten()
    ml_subtype = ml_subtype.flatten()
    ml_stage = ml_stage.flatten()

    if true_order_matrix is not None and true_subtype_assignments is not None:
        tau, subtype_acc, mean_stage_healthy = utils.get_final_metrics(
            true_order_matrix=np.array(true_order_matrix),
            best_order_matrix=best_order_matrix_em,
            true_subtype_assignments=np.array(true_subtype_assignments),
            ml_subtype=ml_subtype_em,
            ml_stage=ml_stage_em,
            diseased_mask=diseased_mask
        )
        end_time = time.time()  

        tau_mcmc, subtype_acc_mcmc, mean_stage_healthy_mcmc = utils.get_final_metrics(
            true_order_matrix=np.array(true_order_matrix),
            best_order_matrix=best_order_matrix_mcmc,
            true_subtype_assignments=np.array(true_subtype_assignments),
            ml_subtype=ml_subtype,
            ml_stage=ml_stage,
            diseased_mask=diseased_mask
        )

        best_order_matrix_em = np.argsort(ml_from_pickle, axis = 1) # n_subtypes, n_biomarkers
        best_order_matrix_mcmc = np.argsort(best_sample, axis = 1)
        tau_argsort = get_tau(np.array(true_order_matrix), best_order_matrix_em)
        tau_argsort_mcmc = get_tau(np.array(true_order_matrix), best_order_matrix_mcmc)
    
    results = {
        "runtime": end_time - start_time,
        "First == Best": np.array_equal(first_sample, best_sample),
        "First == Pickle ML": np.array_equal(first_sample, ml_from_pickle),
        "Best == Pickle ML": np.array_equal(best_sample, ml_from_pickle),
        'tau': tau,
        'tau_argsort': tau_argsort,
        'subtype_acc': subtype_acc,
        'mean_stage_healthy': mean_stage_healthy,
        'tau_mcmc': tau_mcmc,
        'tau_argsort_mcmc': tau_argsort_mcmc,
        'subtype_acc_mcmc': subtype_acc_mcmc,
        'mean_stage_healthy_mcmc': mean_stage_healthy_mcmc,
        'true_n_subtypes': true_n_subtypes,
    }

    if estimate_n_subtypes:
        # -------------------------------------------------------------------------
        # Getting the optimal number of subtypes
        # -------------------------------------------------------------------------
        N_S_max = max_n_subtypes  # Maximum number of subtypes (since you mentioned 2 orderings)
        sustain_model = MixtureSustain(
            L_yes, 
            L_no, 
            biomarker_labels,  # biomarker labels
            N_startpoints, 
            N_S_max, 
            N_iterations_MCMC, 
            output_dir, 
            dataset_name, 
            use_parallel_startpoints
        )
        # Stratified CV folds
        cv = StratifiedKFold(n_splits=SUSTAIN_EVAL_N_FOLD, shuffle=True, random_state=random_state)
        test_idxs = [test.astype(int) for _, test in cv.split(data, target)]

        # Perform cross-validation
        CVIC, _ = sustain_model.cross_validate_sustain_model(test_idxs)
        ml_n_subtypes = utils.choose_optimal_subtypes(CVIC)

        end_time_eval = time.time()
        results.update({
            'runtime_cross_validation': end_time_eval - end_time,
            'ml_n_subtypes': ml_n_subtypes,
            'correct_bool': int(true_n_subtypes == ml_n_subtypes),
            'absolute_error': abs(true_n_subtypes - ml_n_subtypes),
            'relative_error': ml_n_subtypes - true_n_subtypes
            })

    with open(f"{RESULTS_DIR}/{filename}_results.json", "w") as f:
        json.dump(utils.convert_np_types(results), f, indent=4)

def eval_pysubebm(
    filename:str,
    data_file: str,
    n_subtypes: int,
    true_order_matrix: np.ndarray,
    true_subtype_assignments: np.ndarray,
    output_dir: str,
    n_iter: int, 
    burn_in: int,
    thinning: int,
    seed: int,
    save_results: bool = False,  
    max_n_subtypes:int=5,
    n_shuffle: int=2,
    n_subtype_shuffle: int=2, 
    prior_n:float=1.0,
    prior_v:float=1.0,
    N_FOLDS:int=5,
    with_labels:bool=False,
    estimate_n_subtypes:bool=False,
    z_score_norm:bool=True,
):
    results_folder = os.path.join(output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results, _, _, _, _, _, _ = run_bebms(
        data_file= data_file,
        n_subtypes=n_subtypes,
        true_order_matrix=true_order_matrix,
        true_subtype_assignments=true_subtype_assignments,
        output_dir=output_dir,
        n_iter=n_iter,
        n_shuffle=n_shuffle,
        n_subtype_shuffle=n_subtype_shuffle,
        burn_in=burn_in,
        thinning=thinning,
        seed = seed,
        save_results=save_results,
        obtain_results = True,
        with_labels=with_labels,
        save_plots =False,
        z_score_norm=z_score_norm
    )

    if estimate_n_subtypes:

        start_time = time.time()
        _, ml_n_subtypes = cross_validatation(
            data_file=data_file,
            iterations=n_iter,
            n_shuffle=n_shuffle,
            n_subtype_shuffle=n_subtype_shuffle,
            burn_in=burn_in,
            prior_n= prior_n,
            prior_v=prior_v,
            max_n_subtypes=max_n_subtypes,
            N_FOLDS=N_FOLDS,
            seed=seed,
            with_labels=with_labels,
            z_score_norm=z_score_norm
        )
        end_time = time.time()

        results.update({
            'true_n_subtypes': n_subtypes,
            'ml_n_subtypes': ml_n_subtypes,
            'correct_bool': int(n_subtypes == ml_n_subtypes),
            'absolute_error': abs(n_subtypes - ml_n_subtypes),
            'relative_error': ml_n_subtypes - n_subtypes,
            'runtime_cross_validation': end_time-start_time,
        })
    with open(f"{results_folder}/{filename}_results.json", "w") as f:
        json.dump(utils.convert_np_types(results), f, indent=4)