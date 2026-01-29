import pickle

training_dataset_path_list = [
    # 'data_cvrp_10_150_256_withbaseline.pkl',
                              'cvrplib_test_set_A.pkl',
                              # 'cvrp_uniform_cap30-60_n30-60_num10_test.pkl',
                              ]

loaded_datasets = []
for dataset in training_dataset_path_list:
    loaded_dataset_one = pickle.load(open(dataset, 'rb'))
    loaded_datasets.extend(loaded_dataset_one)
    print(f"load dataset from {dataset}")

instance_set = {
    info_id: info
    for info_id, info in enumerate(loaded_datasets)
}

instance_id_set = tuple(instance_set.keys())