'''
Script to run GCN model for Cora node classification
'''

from Dataset_Loader_Node_Classification import Dataset_Loader
from Method_GCN_cora import Method_GCN_Node_Classifier
from Result_Saver import Result_Saver
from Setting_Train_Test_Split import Setting_Train_Test_Split
from Evaluate_Metrics import Evaluate_Metrics

import numpy as np
import torch

if __name__ == '__main__':
    # ---- parameter section ------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- object initialization section -------------------
    dataset_obj = Dataset_Loader(dName='cora', dDescription='Cora citation network')
    dataset_obj.dataset_name = 'cora'
    dataset_obj.dataset_source_folder_path = './stage_5_data/cora/'  # ensure the path includes link and node file.
    data = dataset_obj.load()
    
    # obtain the graph and train_val_test data
    graph = data['graph']
    train_val_test = data['train_test_val']

    method_obj = Method_GCN_Node_Classifier('GCN', '')
    method_obj.set_data(graph, train_val_test)

    result_obj = Result_Saver('GCN_Cora', '')
    result_obj.result_destination_folder_path = './result/cora_result/'
    result_obj.result_destination_file_name = 'cora_gcn_result'

    setting_obj = Setting_Train_Test_Split('Cora', 'GCN for Cora Node Classification')

    evaluate_obj = Evaluate_Metrics()

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print("\nEvaluation Results:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    print('************ Finish ************')
