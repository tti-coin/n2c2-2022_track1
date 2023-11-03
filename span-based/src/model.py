import dataclasses

import torch
import torch.nn as nn
import transformers


@dataclasses.dataclass
class ModelConfig:
    bert_model_path: str
    dropout_rate: float
    bert_layers: int
    window_token: int


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.config = config
        self.window_token = self.config.window_token
        self.bert_config = transformers.AutoConfig.from_pretrained(config.bert_model_path)
        self.bert_config.update({"output_hidden_states": True})
        self.bert = transformers.AutoModel.from_pretrained(config.bert_model_path, config=self.bert_config)
        self.input_dim = self.bert.config.hidden_size
        self.class_dim = 23

        self.fc_all = torch.nn.Linear(self.input_dim, self.class_dim)
        self.bert._init_weights(self.fc_all)
        self.all = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            self.fc_all,
        )

    def forward(self, token_ids: torch.Tensor, token_ids_mask: torch.Tensor, s_tok_idxs: torch.Tensor, e_tok_idxs: torch.Tensor) -> torch.Tensor:
        bert_outputs = self.bert(token_ids, attention_mask=token_ids_mask)

        # Use last n layers
        seq_outputs = [bert_outputs["hidden_states"][-1 * i][:, :].unsqueeze(1) for i in range(1, self.config.bert_layers + 1)]
        t_seq_outputs = torch.cat(seq_outputs, dim=1)
        t_seq_outputs = torch.mean(t_seq_outputs, dim=1)
        span_feats = list()

        # Create span feature
        for seq, s_tok_idx, e_tok_idx in zip(t_seq_outputs, s_tok_idxs, e_tok_idxs):
            mean_feat = torch.mean(seq[max(1, s_tok_idx - self.window_token) : min(seq.size()[0] - 1, e_tok_idx + self.window_token), :], 0)
            s_feat = seq[max(1, s_tok_idx - self.window_token)].unsqueeze(0)
            e_feat = seq[min(seq.size()[0] - 1, e_tok_idx + self.window_token)].unsqueeze(0)
            span_feat = torch.mean(torch.cat((s_feat, mean_feat.unsqueeze(0), e_feat), 0), 0)
            span_feat = torch.cat([span_feat], dim=0).unsqueeze(0)
            span_feats.append(span_feat)

        span_fetas = torch.cat(span_feats, 0)
        outputs = self.all(span_fetas)
        return outputs
