from tools.functions import *
import glob
import pandas as pd

# chose the conditions to combine the individual runs

condition = "sparse_015_015_neurogenesis"
print("Combining data for condition: " + condition)

use_threshold = False
threshold = 0.5

# Path to the data
path = "scripts/output.nosync/"

list_condition = glob.glob(path + "*" + condition + ".pkl")

idx = []
for i in range(len(list_condition)): 
    if list_condition[i][len(path):].split("-")[1][:-4] == condition:
        idx.append(i)
list_condition = np.array(list_condition)[idx].copy()

print("Found " + str(len(list_condition)) + " matching files.")

print("Combining the data...")

# Path to save the data
saving_path = "scripts/final_data/"
off_time = 1
#threshold = [100, 150, 200] # not used if binarize = False
binarize = False # should be false
normalize = True # should be true 

rounding = 4 # rounding for the hamming distance since some of the unique values are very close to each other
rounding_ed = 4 # rounding for the euclidean distance since some of the unique values are very close to each other

aggregated_outputs = {}
switch = True

for version in list_condition: 
    output = load_output(version, path="", extension="")
    Mv_loaded = output["Mv"]
    Mh_loaded = output["Mh"]
    time_test_on = output["time_test_on"]
    time_points_dict = output["time_points_dict"]
    n_seed_patterns = output["n_seed_patterns"]
    n_prototype_per_seed = output["n_prototype_per_seed"]
    n_variations_per_prototype = output["n_variations_per_prototype"]
    after_split_n_per_prototype_test = output["after_split_n_per_prototype_test"]
 
    # Memory interference
    hamming_distances, percent_match, originals, recovered, euclidean_distances, percent_match_ed = hamming_distances_test(Mv_loaded, time_test_on, time_points_dict, off_time=off_time, normalize=normalize,
                                                                                    binarize = binarize, threshold = threshold)
    
    hd_table_between = np.array(hamming_distances[:10 * after_split_n_per_prototype_test * n_seed_patterns]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    pm_table_between = np.array(percent_match[:10 * after_split_n_per_prototype_test * n_seed_patterns]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    pm_table_between_ed = np.array(percent_match_ed[:10 * after_split_n_per_prototype_test * n_seed_patterns]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    hd_table_within = np.array(hamming_distances[10 * after_split_n_per_prototype_test * n_seed_patterns:]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    pm_table_within = np.array(percent_match[10 * after_split_n_per_prototype_test * n_seed_patterns:]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    pm_table_within_ed = np.array(percent_match_ed[10 * after_split_n_per_prototype_test * n_seed_patterns:]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    ed_table_between = np.array(euclidean_distances[:10 * after_split_n_per_prototype_test * n_seed_patterns]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)
    ed_table_within = np.array(euclidean_distances[10 * after_split_n_per_prototype_test * n_seed_patterns:]).reshape(10, after_split_n_per_prototype_test * n_seed_patterns)


    # Pattern separation
    Ids = output["Ids"]
    Ids_normalized = normalizer(Ids)

    time_test_off_loaded = output["time_test_off"]
    t_on = np.setdiff1d(np.arange(1, max(time_test_off_loaded)), time_test_off_loaded)

    n_seed_patterns = output["n_seed_patterns"]
    n_prototype_per_seed = output["n_prototype_per_seed"]
    n_variations_per_prototype = output["n_variations_per_prototype"]
    after_split_n_per_prototype_train = output["after_split_n_per_prototype_train"]
    after_split_n_per_prototype_test = output["after_split_n_per_prototype_test"]    
    off_time = output["off_time"] 

    len_phase = after_split_n_per_prototype_train * n_seed_patterns + after_split_n_per_prototype_test * n_seed_patterns 
    idx_last_phase_plus_testing = len_phase * (n_variations_per_prototype - 1) # -1 means that the the last phase is included in the evaluation. -2 would mean the last two phases are included in the evaluation
    last_phase_plus_testing = t_on[idx_last_phase_plus_testing:]

    inputs_normalized = normalizer(normalizer(Ids[last_phase_plus_testing])+1)
    hidden_normalized = []

    for i in last_phase_plus_testing:
        t_start = time_points_dict["T"+str(i)+"_s"]
        t_stop = time_points_dict["T"+str(i)+"_e"]
        normalized = normalizer(spike_histogram(Mh_loaded, t_start=t_start, t_stop=t_stop).T[1])
        if use_threshold == True:
            normalized[normalized < threshold] = 0
            normalized[normalized >= threshold] = 1
        hidden_normalized.append(normalized)

    #inputs_normalized = normalizer(inputs)
    #hidden_normalized = normalizer(hidden)

    hamming_distances_in = np.zeros((len(inputs_normalized), len(inputs_normalized)))   # actally l1 norm
    ed_in = np.zeros((len(inputs_normalized), len(inputs_normalized)))                  # actually l2 norm

    for i in range(len(inputs_normalized)):
        for j in range(len(inputs_normalized)):
            hamming_distances_in[i,j] = calculate_hamming_distance(inputs_normalized[i], inputs_normalized[j])
            ed_in[i, j] = calculate_eucledian_distance(inputs_normalized[i], inputs_normalized[j])

    hamming_distances_out = np.zeros((len(hidden_normalized), len(hidden_normalized)))
    ed_out = np.zeros((len(hidden_normalized), len(hidden_normalized)))

    for i in range(len(hidden_normalized)):
        for j in range(len(hidden_normalized)):
            hamming_distances_out[i,j] = calculate_hamming_distance(hidden_normalized[i], hidden_normalized[j])
            ed_out[i, j] = calculate_eucledian_distance(hidden_normalized[i], hidden_normalized[j])

    unique_x = np.unique(hamming_distances_in.round(rounding))
    unique_x_ed = np.unique(ed_in.round(rounding_ed))

    sorted_unique_y = []
    sorted_unique_y_ed = []
    for i in range(len(unique_x)):
        sorted_unique_y.append(hamming_distances_out[np.where(hamming_distances_in.round(rounding) == unique_x[i])])
        sorted_unique_y_ed.append(ed_out[np.where(ed_in.round(rounding_ed) == unique_x_ed[i])])

    

    # bind together

    if switch == True: 
        aggregated_outputs["hd_table_between"] = hd_table_between
        aggregated_outputs["ed_table_between"] = ed_table_between
        aggregated_outputs["pm_table_between"] = pm_table_between
        aggregated_outputs["pm_table_between_ed"] = pm_table_between_ed
        aggregated_outputs["hd_table_within"] = hd_table_within
        aggregated_outputs["ed_table_within"] = ed_table_within
        aggregated_outputs["pm_table_within"] = pm_table_within
        aggregated_outputs["pm_table_within_ed"] = pm_table_within_ed
        pattern_separation = pd.DataFrame(sorted_unique_y).T
        pattern_separation_ed = pd.DataFrame(sorted_unique_y_ed).T
        switch = False
    else:
        aggregated_outputs["hd_table_between"] = np.concatenate((aggregated_outputs["hd_table_between"], hd_table_between), axis=1)
        aggregated_outputs["ed_table_between"] = np.concatenate((aggregated_outputs["ed_table_between"], ed_table_between), axis=1)
        aggregated_outputs["pm_table_between"] = np.concatenate((aggregated_outputs["pm_table_between"], pm_table_between), axis=1)
        aggregated_outputs["pm_table_between_ed"] = np.concatenate((aggregated_outputs["pm_table_between_ed"], pm_table_between_ed), axis=1)
        aggregated_outputs["hd_table_within"] = np.concatenate((aggregated_outputs["hd_table_within"], hd_table_within), axis=1)
        aggregated_outputs["ed_table_within"] = np.concatenate((aggregated_outputs["ed_table_within"], ed_table_within), axis=1)
        aggregated_outputs["pm_table_within"] = np.concatenate((aggregated_outputs["pm_table_within"], pm_table_within), axis=1)
        aggregated_outputs["pm_table_within_ed"] = np.concatenate((aggregated_outputs["pm_table_within_ed"], pm_table_within_ed), axis=1)
        pattern_separation = pd.concat([pattern_separation, pd.DataFrame(sorted_unique_y).T], axis=0)
        pattern_separation_ed = pd.concat([pattern_separation_ed, pd.DataFrame(sorted_unique_y_ed).T], axis=0)


# save the combined outputs

import pickle
with open(saving_path + condition + "_aggregated_outputs.pkl", 'wb') as f:
    pickle.dump(aggregated_outputs, f)


pd.DataFrame(aggregated_outputs["hd_table_between"]).to_csv(saving_path + condition + "_hd_table_between.csv")
pd.DataFrame(aggregated_outputs["ed_table_between"]).to_csv(saving_path + condition + "_ed_table_between.csv")
pd.DataFrame(aggregated_outputs["pm_table_between"]).to_csv(saving_path + condition + "_pm_table_between.csv")
pd.DataFrame(aggregated_outputs["pm_table_between_ed"]).to_csv(saving_path + condition + "_pm_table_between_ed.csv")
pd.DataFrame(aggregated_outputs["hd_table_within"]).to_csv(saving_path + condition + "_hd_table_within.csv")    
pd.DataFrame(aggregated_outputs["ed_table_within"]).to_csv(saving_path + condition + "_ed_table_within.csv")    
pd.DataFrame(aggregated_outputs["pm_table_within"]).to_csv(saving_path + condition + "_pm_table_within.csv")
pd.DataFrame(aggregated_outputs["pm_table_within_ed"]).to_csv(saving_path + condition + "_pm_table_within_ed.csv")
pattern_separation.to_csv(saving_path + condition + "_pattern_separation.csv")
pattern_separation_ed.to_csv(saving_path + condition + "_pattern_separation_ed.csv")
pd.DataFrame(unique_x).to_csv(saving_path + condition + "_unique_x.csv")
pd.DataFrame(unique_x_ed).to_csv(saving_path + condition + "_unique_x_ed.csv")


print("Done!")