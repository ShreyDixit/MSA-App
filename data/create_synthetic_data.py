from itertools import product
import pandas as pd
import numpy as np

def generate_synthetic_data(num_patients=300, num_brain_regions=25):
    mean_alteration = 5
    std_alteration = 10

    # Generate brain region alterations and NIHSS scores
    alterations = np.clip(np.random.normal(mean_alteration, std_alteration, (num_patients, num_brain_regions)), 0, 100)
    nihss_scores = np.random.randint(0, 9, num_patients)

    # Adjust alterations to introduce correlations with NIHSS scores
    for i in range(num_brain_regions):
        e = (i + 1) / num_brain_regions * 50
        alterations[:, i] += nihss_scores + np.random.uniform(-e, e, num_patients)
        alterations[:, i] = np.clip(alterations[:, i], 0, 100)

    return alterations, nihss_scores

def generate_synthetic_data_full_msa(num_brain_regions=12):
    # Generate all possible combinations of 1s and 0s for ROIs
    combinations = list(product([0, 1], repeat=num_brain_regions))
    alterations = np.array(combinations)
    
    # Define a simple relationship between the sum of alterations and NIHSS scores
    # For example, scale the sum of 1s to a range of NIHSS scores
    max_nihss_score = 8  # Assuming the max NIHSS score is 8
    min_nihss_score = 0  # Assuming the min NIHSS score is 0
    nihss_scores = np.array([np.interp(np.sum(row), [0, num_brain_regions], [min_nihss_score, max_nihss_score]) for row in alterations]).astype(int)
    
    return alterations, nihss_scores

def save_data(roi_data, nihss_scores, num_voxels, roi_data_filename, nihss_scores_filename, num_voxels_filename):
    roi_data.to_csv(roi_data_filename, index=False)
    nihss_scores.to_csv(nihss_scores_filename, index=False)
    num_voxels.to_csv(num_voxels_filename, header=False)  # Assuming you want to save without header

# Main script
if __name__ == "__main__":
    num_patients = 300
    num_brain_regions = 25
    full_msa = False  # Note: Be cautious with this value; 2^25 generates over 33 million combinations

    if full_msa:
        alterations, nihss_scores_array = generate_synthetic_data_full_msa(num_brain_regions)
    else:
        alterations, nihss_scores_array = generate_synthetic_data(num_patients, num_brain_regions)
    
    # Creating DataFrame for ROI data and NIHSS scores
    columns = [f'ROI_{i+1}' for i in range(num_brain_regions)] + ['NIHSS']
    synthetic_data = pd.DataFrame(np.hstack((alterations, nihss_scores_array.reshape(-1, 1))), columns=columns)
    
    roi_data = synthetic_data.drop('NIHSS', axis=1)
    nihss_scores_df = synthetic_data[['NIHSS']]
    
    # Generating and saving the number of voxels per ROI
    num_voxels = pd.Series(np.clip(np.random.normal(10000, 5000, num_brain_regions).astype(int), 1000, None), index=roi_data.columns)

    # File names
    if full_msa:
        roi_data_filename = 'roi_data_full_msa.csv'
        nihss_scores_filename = 'nihss_scores_full_msa.csv'
        num_voxels_filename = 'num_voxels_full_msa.csv'
    else:
        roi_data_filename = 'roi_data.csv'
        nihss_scores_filename = 'nihss_scores.csv'
        num_voxels_filename = 'num_voxels.csv'
    
    save_data(roi_data, nihss_scores_df, num_voxels, roi_data_filename, nihss_scores_filename, num_voxels_filename)

