import sys
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, img_vect_size, question_vect_size, answer_vect_size):
        super(Net, self).__init__()
        self.fcq=nn.Linear(question_vect_size,answer_vect_size)
        self.fcv=nn.Linear(img_vect_size,answer_vect_size)

    def forward(self,v,q):
        x=self.fcv(v)+self.fcq(q)
        return x

