import pydantic

import torch
import transformers

class ModelConfig(pydantic.BaseModel):
    bert_name:str
    dropout_output:float
    output_dim:int = 5 # BILOU
    id_to_label:dict = pydantic.Field(default_factory=lambda:{0:"Entity"})

class Model(torch.nn.Module):
    def __init__(self, model_config:ModelConfig):
        super(Model, self).__init__()
        self.model_config = model_config.model_copy(deep=True)
        self.bert = transformers.AutoModel.from_pretrained(self.model_config.bert_name)
        self.dropout_output = torch.nn.Dropout(self.model_config.dropout_output)
        self.fc_output = torch.nn.Linear(self.bert.config.hidden_size, self.model_config.output_dim)
        self.bert._init_weights(self.fc_output)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state # [B, seq_len, dim]
        h = self.dropout_output(h) # [B, seq_len, dim]
        logits = self.fc_output(h) # [B, seq_len, output_dim]
        return logits

