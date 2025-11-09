from bebms import generate
import numpy as np 
import json 
import re 
import os 
import yaml

def extract_components(filename):
    pattern = r'^j(\d+)_r([\d.]+)_E(.*?)_m(\d+)$'
    match = re.match(pattern, filename)
    if match:
        return match.groups()  # returns tuple (J, R, E, M)
    return None

def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def convert_np_types(obj):
    """Convert numpy types in a nested dictionary to Python standard types."""
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np_types(obj.tolist())
    else:
        return obj

if __name__ == '__main__':
    OUTPUT_DIR = 'data'

    # Get path to default parameters
    # params_file = get_params_path()
    # params_file = 'adni_params_ucl_gmm_large_effect_sizes.json'
    params_file = 'adni_params_ucl_gmm.json'
    # params_file = 'adni_params_cp.json'

    with open(params_file) as f:
        params = json.load(f)

    config = load_config()
    print("Loaded config:")
    # print(json.dumps(config, indent=4))

    rng = np.random.default_rng(config['GEN_SEED'])

    ########################################################################
    # Generate data
    ########################################################################
    all_exp_dicts = []
    for exp_name in config['EXPERIMENT_NAMES']:
        random_state = rng.integers(0, 2**32 - 1)
        exp_dict = generate(
            experiment_name = exp_name,
            # params_file=params_file,
            params=params,
            js = config['JS'],
            rs = config['RS'],
            num_of_datasets_per_combination=config['N_VARIANTS'],
            output_dir=OUTPUT_DIR,
            seed=random_state,
            keep_all_cols = False,
            temperature_lo=config['TEMPERATURE_LO'],
            temperature_hi=config['TEMPERATURE_HI'],
            n_sub_lo=config['N_SUB_LO'],
            n_sub_hi=config['N_SUB_HI'],
            subtype_dirichlet_priors=config['GEN_DIRICHLET_PRIORS'],
            subtype_length_lo=config['SUBTYPE_LENGTH_LO'],
            fixed_biomarker_order=True,
        )
        all_exp_dicts.append(exp_dict)

        # flatten the dictionaries
        combined = {k: v for d in all_exp_dicts for k, v in d.items()}
        # convert numpy types to python standards types in order to save to json
        combined = convert_np_types(combined)

        # Dump the JSON
        with open(f"true_order_and_stages.json", "w") as f:
            json.dump(combined, f, indent=2)