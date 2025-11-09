import yaml
import os 

def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config = load_config()
    js = config['JS']
    rs = config['RS']
    m_range = config['N_VARIANTS']
    experiment_names = config['EXPERIMENT_NAMES']
    
    # exp_name = experiment_names[-1]
    res = []
    for m in range(m_range):
        for exp_name in experiment_names:
            for j in js:
                for r in rs:
                    res.append(f"j{j}_r{r}_E{exp_name}_m{m}")
                    
    # Write the results to a text file
    with open('all_combinations.txt', 'w') as file:
        for line in res:
            file.write(f"{line}\n")