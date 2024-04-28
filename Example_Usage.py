from Attention import Attention
#%%
# İnitiate the class
attent=Attention(0)
#%%
attent.soft_attention("this is a test")
attent.print_vector_to_json()
#%%
import torch
from torch import nn

class TextProcessingModel(nn.Module):
    def __init__(self, attention_dim,num_classes, glove_path):
        super(TextProcessingModel, self).__init__()
        self.attention_layer = Attention(dim=attention_dim, glove_path=glove_path)
        self.output_layer = nn.Linear(attention_dim, num_classes)

    def forward(self, text):
        attention_output = self.attention_layer(text, attention_type="soft")
        attention_mean = torch.mean(torch.stack(attention_output), dim=0)
        output = self.output_layer(attention_mean)
        return output


model = TextProcessingModel(attention_dim=25,num_classes=2, glove_path="glove.twitter.27B.25d.txt")

#%%
