import pandas as pd 
import numpy as np 
from typing import List, Dict, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt 

def get_adni_filtered(
        raw:str, 
        meta_data:List[str], 
        select_biomarkers:List[str], 
        diagnosis_list:List[str]
    ) -> pd.DataFrame:
    """Get the filtered data. 
    meta_data = ['PTID', 'DX_bl', 'VISCODE', 'COLPROT']

    select_biomarkers = ['MMSE_bl', 'Ventricles_bl', 'WholeBrain_bl', 
                'MidTemp_bl', 'Fusiform_bl', 'Entorhinal_bl', 
                'Hippocampus_bl', 'ADAS13_bl', 'PTAU_bl', 
                'TAU_bl', 'ABETA_bl', 'RAVLT_immediate_bl'
    ]

    diagnosis_list = ['CN', 'EMCI', 'LMCI', 'AD']
    """
    df = pd.read_csv(raw, usecols=meta_data + select_biomarkers, low_memory=False)
    # 2. Filter to baseline and known diagnoses
    df = df[df['VISCODE'] == 'bl']
    df = df[df['DX_bl'].isin(diagnosis_list)]

    # 3. Convert biomarker columns to numeric (handles garbage strings like '--')
    for col in select_biomarkers:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop rows with any NaN in biomarkers
    df = df.dropna(subset=select_biomarkers).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    print(len(df))
    if len(df.PTID.unique()) == len(df):
        print('No duplicates!')
    else:
        print('Data has duplicates!')
    
    # Print DX distribution
    counts = Counter(df['DX_bl'])
    total = sum(counts.values())

    for k, v in counts.items():
        perc = 100 * v / total
        print(f"{k}: {v} ({perc:.1f}%)")
    
    print('----------------------------------------------------')
    
    # Print Cohort distribution
    counts = Counter(df['COLPROT'])
    total = sum(counts.values())

    for k, v in counts.items():
        perc = 100 * v / total
        print(f"{k}: {v} ({perc:.1f}%)")

    return df 

def process_data(
        df:pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
    """To get the required output for debm, ucl, and sa-ebm
    df: adni_filtered
    """
    df.columns = df.columns.str.replace('_bl', '', regex=False)

    # ICV normalization because brain sizes vary a lot 
    df['VentricleNorm']  = df['Ventricles']  / df['ICV']
    df['HippocampusNorm'] = df['Hippocampus'] / df['ICV']
    df['WholeBrainNorm']  = df['WholeBrain']  / df['ICV']
    df['EntorhinalNorm']  = df['Entorhinal']  / df['ICV']
    df['FusiformNorm']    = df['Fusiform']    / df['ICV']
    df['MidTempNorm']     = df['MidTemp']     / df['ICV']

    df.drop([
        'VISCODE', 'COLPROT', 'ICV', 'Ventricles', 
        'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'PTID'
    ], axis=1, inplace=True)

    dx_array = df.DX.to_numpy()

    df['diseased'] = [int(dx != 'CN') for dx in df.DX]

    df.drop(['DX'], axis=1, inplace=True)

    output_df = df.copy()
    output_df['participant'] = ""
    
    # Ordered biomarkers, to match the ordering outputs later
    # we don't need the last 'diseased' column
    ordered_biomarkers = df.columns[:-1].to_numpy()
    # for ucl
    data_matrix = df.to_numpy()
    return output_df, data_matrix, ordered_biomarkers, dx_array


def save_debm_heatmap(
    bootstrap_orderings: List,  # List of orderings from bootstrap iterations
    biomarker_names: List[str],
    folder_name: str,
    file_name: str,
    title: str,
    mean_ordering: Optional[List[int]] = None,
    plot_mean_order: bool = True  # Whether to sort by mean ordering
):
    """
    Create a heatmap showing the positional variance of biomarkers across bootstrap iterations.
    Similar to what's shown in Archetti 2019 and the pyebm documentation.
    
    Args:
        bootstrap_orderings: List of orderings from each bootstrap iteration
        biomarker_names: List of biomarker names
        folder_name: Directory to save the plot
        file_name: Name of the output file
        title: Title for the plot
        mean_ordering: Mean ordering across all bootstraps (if not provided, will be calculated)
        plot_mean_order: If True, sort biomarkers by their mean position
    """
    os.makedirs(folder_name, exist_ok=True)
    
    n_biomarkers = len(biomarker_names)
    n_bootstraps = len(bootstrap_orderings)
    
    # Create a matrix to count occurrences of each biomarker at each position
    position_counts = np.zeros((n_biomarkers, n_biomarkers))
    
    # Count how many times each biomarker appears at each position
    for ordering in bootstrap_orderings:
        for position, biomarker_idx in enumerate(ordering):
            position_counts[biomarker_idx, position] += 1
    
    # Convert to DataFrame
    biomarker_position_df = pd.DataFrame(
        position_counts,
        index=biomarker_names,
        columns=range(1, n_biomarkers + 1)  # Stage positions 1 to N
    )
    
    # If mean ordering is provided or we want to plot by mean order
    if plot_mean_order:
        
        # Reorder biomarkers by mean ordering
        ordered_biomarker_names = [biomarker_names[i] for i in mean_ordering]
        biomarker_position_df = biomarker_position_df.loc[ordered_biomarker_names]
        
        # Add ordering numbers to biomarker names
        renamed_index = [f"{name} ({i+1})" for i, name in enumerate(ordered_biomarker_names)]
        biomarker_position_df.index = renamed_index
    
    # Normalize to show proportions/probabilities
    biomarker_position_df = biomarker_position_df.div(n_bootstraps)
    
    # Find the longest biomarker name
    max_name_length = max(len(name) for name in biomarker_position_df.index)
    
    # Dynamically adjust figure size
    fig_width = max(10, min(20, n_biomarkers * 0.5))  # Scale with number of biomarkers
    fig_height = max(8, min(15, n_biomarkers * 0.4))
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(
        biomarker_position_df,
        annot=True,
        cmap="Blues",  # Use Blues to match pyebm visualization
        linewidths=0.5,
        cbar_kws={'label': 'Probability'},
        fmt=".2f",
        vmin=0,
        vmax=1
    )
    
    plt.xlabel('Stage Position')
    plt.ylabel('Biomarker')
    plt.title(title)
    
    # Adjust y-axis ticks
    plt.yticks(rotation=0, ha='right')
    
    # Adjust margins
    plt.subplots_adjust(left=0.3 if max_name_length > 20 else 0.2)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{folder_name}/{file_name}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{folder_name}/{file_name}.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    return biomarker_position_df