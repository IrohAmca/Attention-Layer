import torch
from torch.nn import Softmax, Module, Linear
import json

class Attention(Module):
    def __init__(self, dim, glove_path="glove.twitter.27B.25d.txt"):  # The dimensions are length of glove vectors
        super(Attention, self).__init__()

        self.softmax = Softmax(dim=-1)
        self.glove_vectors = self.load_glove_vectors(glove_path)
        self.output = []

        self.Q = Linear(dim, dim)
        self.K = Linear(dim, dim)
        self.V = Linear(dim, dim)

        self.Q_tanh = torch.nn.Tanh()
        self.K_tanh = torch.nn.Tanh()
        self.V_tanh = torch.nn.Tanh()


    def forward(self, text, attention_type="soft", temperature=1.0):
        if attention_type == "soft":
            self.soft_attention(text)
        elif attention_type == "hard":
            self.hard_attention(text)
        elif attention_type == "temporal":
            self.temporal_attention(text, temperature=temperature)
        elif attention_type == "self":
            self.self_attention(text)
        return self.output

    def load_glove_vectors(self, file_path):
        vectors = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                vectors[word] = vector
        return vectors

    def soft_attention(self, text):
        words = text.split()
        vectors = [self.glove_vectors[word] for word in words if word in self.glove_vectors]
        stacked_vectors = torch.stack(vectors, dim=0)
        transposed_vectors = torch.transpose(stacked_vectors, 0, 1)
        self_attention = self.calculate_attention(stacked_vectors, transposed_vectors)
        self.output = self_attention

    def hard_attention(self, text):
        words = text.split()
        vectors = [self.glove_vectors[word] for word in words if word in self.glove_vectors]
        stacked_vectors = torch.stack(vectors, dim=0)
        transposed_vectors = torch.transpose(stacked_vectors, 0, 1)
        attention_vectors = self.calculate_attention_hard(stacked_vectors, transposed_vectors)
        attention_vectors = torch.stack(attention_vectors, dim=0)
        max_indices = torch.argmax(attention_vectors, dim=1)
        hard_attention_mask = torch.zeros_like(attention_vectors)
        hard_attention_mask[torch.arange(hard_attention_mask.shape[0]), max_indices] = 1
        hard_attention_mask = hard_attention_mask.unsqueeze(-1)
        hard_attention_result = torch.mul(hard_attention_mask, stacked_vectors)
        self.output = hard_attention_result

    def temporal_attention(self, text, temperature):
        words = text.split()
        vectors = [self.glove_vectors[word] for word in words if word in self.glove_vectors]
        stacked_vectors = torch.stack(vectors, dim=0)
        transposed_vectors = torch.transpose(stacked_vectors, 0, 1)
        self_attention = self.calculate_attention(stacked_vectors, transposed_vectors, temperature=temperature)
        self.output = self_attention

    def self_attention(self, text):
        words = text.split()
        vectors = [self.glove_vectors[word] for word in words if word in self.glove_vectors]
        stacked_vectors = torch.stack(vectors, dim=0)  # Change dim to 0
        attention_output = self.trainable_attention(stacked_vectors)
        self.output = attention_output

    def calculate_attention(self, stacked_vectors, transposed_vectors, temperature=None):
        attention_vectors = []
        if temperature is None:
            for vector in stacked_vectors:
                attention_vector = self.softmax(torch.matmul(vector, transposed_vectors))
                attention_vectors.append(attention_vector)
            result = torch.matmul(torch.stack(attention_vectors, dim=0), stacked_vectors)
        else:
            for vector in stacked_vectors:
                attention_vector = self.softmax_with_temperature(torch.matmul(vector, transposed_vectors), temperature)
                attention_vectors.append(attention_vector)
            result = torch.matmul(torch.stack(attention_vectors, dim=0), stacked_vectors)
        return result
    def calculate_attention_hard(self,stacked_vectors,transposed_vectors,temperature=None):
        attention_vectors=[]
        if temperature == None:
            for vector in stacked_vectors:
                attention_vector = self.softmax(torch.matmul(vector, transposed_vectors))
                attention_vectors.append(attention_vector)
            return attention_vectors
        else:
            for vector in stacked_vectors:
                attention_vector = self.softmax_with_temperature(torch.matmul(vector, transposed_vectors),temperature)
                attention_vectors.append(attention_vector)
            return attention_vectors
    def softmax_with_temperature(self, logits, temperature=1.0):
        scaled_logits = logits / temperature
        return self.softmax(scaled_logits)

    def print_vector_to_json(self):
        vector = {str(i): self.output[i].tolist() for i in range(len(self.output))}
        with open('vector.json', 'w') as f:
            json.dump(vector, f)

    def trainable_attention(self, x):
        dim = x.shape[1]
        self.Q = Linear(dim, dim)
        self.K = Linear(dim, dim)
        self.V = Linear(dim, dim)
        queries = self.Q(x)
        queries = self.Q_tanh(queries)

        keys = self.K(x)
        keys = self.K_tanh(keys)

        values = self.V(x)
        values = self.V_tanh(values)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) # When change the dimensions, values of transpose function should be changed
        attention_scores = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_scores, values)
        return attention_output

#%%

#%%
