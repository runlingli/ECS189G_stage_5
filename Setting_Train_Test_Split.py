'''
Concrete Setting class for training and testing using train/test split
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from base_class.setting import setting
import torch
import numpy as np
from collections import defaultdict

class Setting_Train_Test_Split(setting):
    def __init__(self, sName=None, sDescription=None):
        super(Setting_Train_Test_Split, self).__init__(sName, sDescription)

    def sample_train_test(self, labels, num_classes, train_per_class, test_per_class):
        label_to_index = defaultdict(list)
        for i, y in enumerate(labels):
            label_to_index[int(y)].append(i)

        idx_train = []
        idx_test = []

        for c in range(num_classes):
            indices = label_to_index[c]
            np.random.shuffle(indices)
            idx_train += indices[:train_per_class]
            idx_test += indices[train_per_class:train_per_class + test_per_class]

        return torch.LongTensor(idx_train), torch.LongTensor(idx_test)

    def load_run_save_evaluate(self):
        # Step 1: Load data
        print('Loading dataset...')
        data = self.dataset.load()
        graph = data['graph']
        labels = graph['y']
        dataset_name = self.dataset.dataset_name.lower()

        # Step 2: Print dataset info
        print(f"Dataset: {dataset_name}")
        print(f"Number of nodes: {labels.shape[0]}")
        print(f"Feature shape: {graph['X'].shape}")
        print(f"Number of classes: {len(set(labels.tolist()))}")
        print("Label distribution:")
        unique, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique.tolist(), counts.tolist()):
            print(f"  Class {label}: {count} samples")

        # Step 3: Dataset-specific sampling
        if dataset_name == 'cora':
            idx_train, idx_test = self.sample_train_test(labels, 7, 20, 150)
        elif dataset_name == 'citeseer':
            idx_train, idx_test = self.sample_train_test(labels, 6, 20, 200)
        elif dataset_name == 'pubmed':
            idx_train, idx_test = self.sample_train_test(labels, 3, 20, 200)

        # Step 4: Update partition
        train_test = {
            'idx_train': idx_train,
            'idx_test': idx_test,
        }

        # Step 5: Inject into method
        self.method.set_data(graph, train_test)

        # Step 6: Run method
        print('Training and Testing...')
        pred, gt = self.method.run()

        # Step 7: Save result
        print('Saving results...')
        self.result.save()

        # Step 8: Evaluate
        print('Evaluating performance...')
        scores = self.evaluate.evaluate(pred, gt)

        return scores
