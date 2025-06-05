'''
Script to run GCN model for Citeseer node classification
'''

from Dataset_Loader_Node_Classification import Dataset_Loader
from Method_GCN_citeseer import Method_GCN_Node_Classifier
from Result_Saver import Result_Saver
from Setting_Train_Test_Split import Setting_Train_Test_Split
from Evaluate_Metrics import Evaluate_Metrics

import numpy as np
import random
import torch
import sys
from datetime import datetime

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

if __name__ == '__main__':
    # set output file
    output_file = './citeseer_evaluation.txt'
    dual_output = DualOutput(output_file)
    sys.stdout = dual_output
    
    
    # ---- parameter section ------------------------------
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- object initialization section -------------------
    dataset_obj = Dataset_Loader(dName='citeseer', dDescription='Citeseer citation network')
    dataset_obj.dataset_name = 'citeseer'
    dataset_obj.dataset_source_folder_path = './stage_5_data/citeseer/'  # ensure the path includes link and node file.
    data = dataset_obj.load()
    # obtain the graph and train_val_test data
    graph = data['graph']
    train_test = data['train_test_val']
    
    for i in range(50):
        method_obj = Method_GCN_Node_Classifier('GCN', '')
        method_obj.set_data(graph, train_test)

        result_obj = Result_Saver('GCN_Citeseer', '')
        result_obj.result_destination_folder_path = './result/citeseer_result/'
        result_obj.result_destination_file_name = 'citeseer_gcn_result'

        setting_obj = Setting_Train_Test_Split('Citeseer', 'GCN for Citeseer Node Classification')

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
    
    # restore standard output and close file
    sys.stdout = dual_output.terminal
    dual_output.close()
