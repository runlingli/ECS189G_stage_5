'''
Visualization module for deep learning model training process
'''

import matplotlib.pyplot as plt
import os

class Training_Visualizer:
    def __init__(self, save_dir='./'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_progress(self, train_losses, title="Training Progress"):
 
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{title} - Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, title + '_training.png')
        plt.savefig(save_path)
        plt.close() 