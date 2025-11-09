import os
import json
import pandas as pd
import re
import numpy as np 
import yaml
from tqdm import tqdm
import shutil
import bebms.utils as utils 
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def extract_components(filename):
    # filename without "_results.json"
    name = filename.replace('_results.json', '')
    pattern = r'^j(\d+)_r([\d.]+)_E(.*?)_m(\d+)$'
    match = re.match(pattern, name)
    if match:
        return match.groups()  # returns tuple (J, R, E, M)
    return None

def generate_expected_files(config):
    """Generate all expected (algo, filename) tuples based on config"""
    expected = []
    for algo in config['ALGO_NAMES']:
        for J in config['JS']:
            for R in config['RS']:
                for E in config['EXPERIMENT_NAMES']:
                    for M in range(config['N_VARIANTS']):
                        fname = f"j{J}_r{R}_E{E}_m{M}_results.json"
                        expected.append((algo, fname))
    return set(expected)

def main():

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open("true_order_and_stages.json", "r") as f:
        true_order_and_stages = json.load(f)
    
    ALGONAMES = config['ALGO_NAMES']
    OUTPUT_DIR = config['OUTPUT_DIR']
    JS = config['JS']
    RS = config['RS']
    EXPERIMENTS = config['EXPERIMENT_NAMES']
    N_VARIANTS = config['N_VARIANTS']

    rng = np.random.default_rng(42)

    titles = [
        "Exp 1: S & Ordinal kj (DM) & X (Normal)",
        "Exp 2: S & Ordinal kj (DM) & X (Non-Normal)",
        "Exp 3: S & Ordinal kj (Uniform) & X (Normal)",
        "Exp 4: S & Ordinal kj (Uniform) & X (Non-Normal)",
        "Exp 8: S & Continuous kj (Uniform) & X (Sigmoid)",
        "Exp 9: S & Continuous kj (Skewed) & X (Sigmoid)",
        "Exp 5: S & Continuous kj (Uniform) & X (Normal)",
        "Exp 6: S & Continuous kj (Uniform) & X (Non-Normal)",
        "Exp x: S & Continuous kj (Skewed) & X (Normal)",
        "Exp 7: S & Continuous kj (Skewed) & X (Non-Normal)",
        "Exp 10: xi (Normal) & Continuous kj (Skewed) & X (Sigmoid)",
        "Exp 11: xi (Normal) & Continuous kj (Skewed) & X (Normal)"
    ]

    exp_nums = [1,2,3,4,8,9,5,6,12,7,10,11]

    # Normalize mapping dictionaries
    CONVERT_E_DICT = {k: v for k, v in zip(EXPERIMENTS, titles)}
    GET_E_NUM = dict(zip(titles, exp_nums))
    GOOD_ALGONAMES = ['BebmS (Blind)', 'BebmS', 'SuStaIn (GMM)', 'SuStaIn (KDE)']
    CONVERT_ALGO_DICT = dict(zip(ALGONAMES, GOOD_ALGONAMES))

    # Initialize tracking structures
    expected_files = generate_expected_files(config)
    found_files = set()
    missing_files = set()
    failed_files = []
    records = []

    # Process all algorithms
    for algo in tqdm(ALGONAMES, desc="Processing algorithms"):
        algo_dir = os.path.join(OUTPUT_DIR, algo, "results")

        if not os.path.exists(algo_dir):
            print(f"\nWarning: Missing directory for {algo}")
            continue

        # Process all result files
        files = [f for f in os.listdir(algo_dir) if f.endswith('_results.json')]
        for fname in tqdm(files, desc=f"{algo}", leave=False):
            metadata = true_order_and_stages[fname.replace('_results.json', '')]
            n_subtypes = metadata['N_SUB']
            mallows_temperature=metadata['TEMPERATURE']
            kendalls_w = metadata['CONCENTRATION']
            true_orderings = np.array(metadata['TRUE_ORDERINGS'])
            true_subtypes = np.array(metadata['TRUE_SUBTYPE_ASSIGNMENTS'])
            true_stages = np.array(metadata['TRUE_STAGE_ASSIGNMENTS'])
            healthy_mask = (true_stages == 0)
            diseased_mask = (healthy_mask == 0)
            data_size = len(true_subtypes)

            full_path = os.path.join(algo_dir, fname)

            # Track found files
            found_files.add((algo, fname))

            # Parse filename components
            components = extract_components(fname)
            if not components:
                failed_files.append((full_path, "Invalid filename format"))
                continue

            J, R, E, M = components
            try:
                J = int(J)
                R = float(R)
                M = int(M)
            except ValueError:
                failed_files.append((full_path, "Invalid numeric format in filename"))
                continue
                
            # Validate against config
            if J not in JS:
                failed_files.append((full_path, f"Invalid J value {J}"))
                continue
            if R not in RS:
                failed_files.append((full_path, f"Invalid R value {R}"))
                continue
            if E not in EXPERIMENTS:
                failed_files.append((full_path, f"Invalid experiment {E}"))
                continue
            if not (0 <= M < N_VARIANTS):
                failed_files.append((full_path, f"Invalid M value {M}"))
                continue

            # Load and validate JSON content
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                
                # if 'kendalls_tau' not in data or 'mean_absolute_error' not in data:
                #     failed_files.append((full_path, "Missing metrics in JSON"))
                #     continue

                # algo_pretty = CONVERT_ALGO_DICT.get(algo.lower(), algo)  # fallback to raw if not found
                algo_pretty = CONVERT_ALGO_DICT.get(algo, algo)
                E_pretty = CONVERT_E_DICT.get(E, E)
                E_num = GET_E_NUM.get(E_pretty, 0)

                subtype_acc = None
                mean_stage_healthy = None
                runtime_cross_validation = None 
                absolute_error_n_subtypes = None 
                relative_error = 100
                correct_bool = 100
                runtime = 0.0

                if algo == 'pysubebm':
                    temp_algo = 'Random Guessing'
                    estimated_orderings = np.array([rng.permutation(np.arange(12)) for _ in range(n_subtypes)])
                    ml_subtype = rng.integers(1, n_subtypes+1, size=data_size) 
                    ml_stage = rng.integers(0, 13, size=data_size)
                    n = len(estimated_orderings)
                    dist = np.zeros((n, n))
                    # i can safely use the sequence results because they are the indices of the fixed input biomarker array!
                    for i in range(n):
                        for j in range(n):
                            dist[i,j]= utils.normalized_kendalls_tau_distance(
                                true_orderings[i], estimated_orderings[j])
                        
                    # This finds the best matching: estimated_indices[i] -> true_indices[i]
                    estimated_indices, true_indices = linear_sum_assignment(dist)
                    # Calculate the matched Kendall's Tau
                    kendalls_tau = dist[estimated_indices, true_indices].mean()

                    ml_subtypes = ml_subtype[diseased_mask]
                    true_subtype_assignments = true_subtypes[diseased_mask]
                    subtype_acc = adjusted_rand_score(true_subtype_assignments, ml_subtypes)
                    mean_stage_healthy = np.mean(ml_stage[healthy_mask])
                    estimated_n_subtype = rng.integers(1, 7, size = 1)
                    absolute_error_n_subtypes = abs(estimated_n_subtype[0] - n_subtypes)

                    records.append({
                        'J': J,
                        'R': R,
                        'E': E_pretty,
                        'M': M,
                        'E_Num': int(E_num),
                        'algo': temp_algo,
                        'n_subtypes': n_subtypes,
                        'mallows_temperature': mallows_temperature,
                        'kendalls_w': kendalls_w,
                        'runtime': runtime,
                        'kendalls_tau': kendalls_tau,
                        'subtype_acc': subtype_acc,
                        'mean_stage_healthy': mean_stage_healthy,
                        'runtime_cross_validation': runtime_cross_validation,
                        'absolute_error_n_subtypes': absolute_error_n_subtypes,
                        'relative_error': relative_error,
                        'correct_bool': correct_bool
                    })
                if 'pysubebm' in algo:
                    kendalls_tau = data['kendalls_tau']
                    subtype_acc = data['subtype_acc']
                    mean_stage_healthy = data['mean_stage_healthy']
                else:
                    kendalls_tau = data['tau_argsort']
                    subtype_acc = data['subtype_acc_mcmc']
                    mean_stage_healthy = data['mean_stage_healthy_mcmc']
                runtime = data['runtime']/60
                if E_num == 1:
                    runtime_cross_validation = data['runtime_cross_validation']
                    absolute_error_n_subtypes = data['absolute_error']
                    relative_error = data['relative_error']
                    correct_bool = data['correct_bool']

                records.append({
                    'J': J,
                    'R': R,
                    'E': E_pretty,
                    'M': M,
                    'E_Num': int(E_num),
                    'algo': algo_pretty,
                    'n_subtypes': n_subtypes,
                    'mallows_temperature': mallows_temperature,
                    'kendalls_w': kendalls_w,
                    'runtime': runtime,
                    'kendalls_tau': kendalls_tau,
                    'subtype_acc': subtype_acc,
                    'mean_stage_healthy': mean_stage_healthy,
                    'runtime_cross_validation': runtime_cross_validation,
                    'absolute_error_n_subtypes': absolute_error_n_subtypes,
                    'relative_error': relative_error,
                    'correct_bool': correct_bool
                })
            except json.JSONDecodeError:
                failed_files.append((full_path, "Invalid JSON format"))
            except Exception as e:
                failed_files.append((full_path, f"Unexpected error: {str(e)}"))
    
    # Calculate missing files
    missing_files = expected_files - found_files

    # Save results
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(by=['J', 'R', 'E', 'M', 'algo'])
        df.to_csv('all_results.csv', index=False)
        print(f"\nSaved {len(df)} valid records to all_results.csv")

    # Save diagnostics
    if missing_files:
        unique_missing_fnames = set([x[1].replace("_results.json", "") for x in missing_files])
        with open('missing_files.txt', 'w') as f:
            f.write("Algorithm, Filename\n")
            for algo, fname in sorted(missing_files):
                f.write(f"{algo}, {fname}\n")
        print(f"Logged {len(missing_files)} missing files to missing_files.txt")

        # Save NA_COMBINATIONS.txt 
        with open('na_combinations.txt', 'w') as f:
            print(f'Number of unique missing fnames: {len(unique_missing_fnames)}')
            for fname in sorted(unique_missing_fnames):
                f.write(f"{fname}\n")
        print(f"Logged {len(unique_missing_fnames)} unique missing files to na_combinations.txt")

        # Copy err and out logs 
        # Create the error_logs directory if it doesn't exist
        if not os.path.exists('error_logs'):
            os.makedirs('error_logs')
        
        ERR_LOGS = [f"eval_{x}.err" for x in unique_missing_fnames]
        OUT_LOGS = [f"eval_{x}.out" for x in unique_missing_fnames]
        LOG_LOGS = [f"eval_{x}.log" for x in unique_missing_fnames]
        # Copy each file from logs to error_logs
        for filename in ERR_LOGS + OUT_LOGS + LOG_LOGS:
            source_path = os.path.join('logs', filename)
            dest_path = os.path.join('error_logs', filename)
            try:
                shutil.copy2(source_path, dest_path)
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        print("Done copying files to error_logs folder")        
        
    if failed_files:
        with open('failed_files.txt', 'w') as f:
            f.write("Path, Reason\n")
            for path, reason in failed_files:
                f.write(f"{path}, {reason}\n")
        print(f"Logged {len(failed_files)} failed files to failed_files.txt")

if __name__ == '__main__':
    main()
