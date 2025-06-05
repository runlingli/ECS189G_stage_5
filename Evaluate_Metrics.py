'''
Concrete Evaluate class for multiple evaluation metrics
'''

from base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluate_Metrics(evaluate):
    def __init__(self, eName=None, eDescription=None):
        super(Evaluate_Metrics, self).__init__(eName, eDescription)

    def evaluate(self, pred_y, true_y):
        if hasattr(true_y, 'detach'):
            true_y = true_y.cpu().tolist()
        if hasattr(pred_y, 'detach'):
            pred_y = pred_y.cpu().tolist()
        
        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='macro')
        recall = recall_score(true_y, pred_y, average='macro')
        f1 = f1_score(true_y, pred_y, average='macro')
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
