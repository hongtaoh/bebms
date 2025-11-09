"""
Use this script to get the adni theta/phi using ucl gmm. 
"""

import numpy as np 
from kde_ebm.mixture_model import fit_all_gmm_models
import utils_adni 
import json 

if __name__ == "__main__":

    meta_data = ['PTID', 'DX_bl', 'VISCODE', 'COLPROT']

    select_biomarkers = ['MMSE_bl', 'Ventricles_bl', 'WholeBrain_bl', 
                'MidTemp_bl', 'Fusiform_bl', 'Entorhinal_bl', 
                'Hippocampus_bl', 'ADAS13_bl', 'PTAU_bl', 
                'TAU_bl', 'ABETA_bl', 'RAVLT_immediate_bl', 'ICV_bl'
    ]

    diagnosis_list = ['CN', 'EMCI', 'LMCI', 'AD']

    raw = '../mlhc_sub/ADNIMERGE.csv'

    adni_filtered = utils_adni.get_adni_filtered(
        raw, meta_data, select_biomarkers, diagnosis_list
    )
    output_df, data_matrix, ordered_biomarkers, dx_array = utils_adni.process_data(df = adni_filtered)
    np.save("dx_array.npy", dx_array)
    output_df.to_csv('adni.csv',index=False)

    data = data_matrix[:, :-1].astype(np.float64)
    target = data_matrix[:, -1].astype(np.int64)

    mixtures = fit_all_gmm_models(data, target)
    # Extract likelihoods for each biomarker
    L_yes = np.zeros(data.shape)
    L_no = np.zeros(data.shape)
    for i in range(data.shape[1]):
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])

    print(L_yes.shape, L_no.shape)  # Should both be (num_subjects, num_biomarkers)
    print("NaN in L_yes:", np.isnan(L_yes).any())
    print("NaN in L_no:", np.isnan(L_no).any())
    print("Min L_yes:", np.min(L_yes))
    print("Min L_no:", np.min(L_no))

    # Create the dictionary directly
    gmm_params = {}
    for i, biomarker_name in enumerate(ordered_biomarkers):
        mixture = mixtures[i]
        gmm_params[biomarker_name] = {
            "theta_mean": float(mixture.ad_comp.mu),
            "theta_std": float(mixture.ad_comp.sigma), 
            "phi_mean": float(mixture.cn_comp.mu),
            "phi_std": float(mixture.cn_comp.sigma)
        }
    
    gmm_params_large_ef = {} # large effect sizes
    multiplier = 1
    divider = 3
    for i, biomarker_name in enumerate(ordered_biomarkers):
        mixture = mixtures[i]
        gmm_params_large_ef[biomarker_name] = {
            "theta_mean": float(mixture.ad_comp.mu * multiplier),
            "theta_std": float(mixture.ad_comp.sigma/divider * multiplier), 
            "phi_mean": float(mixture.cn_comp.mu*multiplier),
            "phi_std": float(mixture.cn_comp.sigma/divider * multiplier)
        }
    
    with open('adni_params_ucl_gmm.json', 'w') as f:
        json.dump(gmm_params, f, indent=4)
    
    with open('adni_params_ucl_gmm_large_effect_sizes.json', 'w') as f:
        json.dump(gmm_params_large_ef, f, indent=4)