import sys
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, question_vect_size, answer_vect_size):
        super(Net, self).__init__()
        self.fc1=nn.Linear(question_vect_size,answer_vect_size)

    def forward(self,x):
        x=self.fc1(x)
        return x

