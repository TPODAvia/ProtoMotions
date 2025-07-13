import torch
from torch import nn
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP
from protomotions.agents.common.transformer import Transformer
import copy


class LightweightStudentModel(nn.Module):
    """
    Lightweight Student Model for distillation.
    
    This model combines all capabilities (path following, goal-directed, steering, masked mimic)
    in a compact architecture that can be efficiently deployed.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create actor and critic with deep copied configs
        actor_config = copy.deepcopy(self.config.actor)
        critic_config = copy.deepcopy(self.config.critic)
        
        self._actor = instantiate(actor_config)
        self._critic = instantiate(critic_config)
        
        self._assert_no_shared_params_or_buffers()

    def _assert_no_shared_params_or_buffers(self):
        """Assert that no parameters or buffers share memory (DDP safety)."""
        seen = {}
        for name, param in self.named_parameters():
            addr = param.data.storage().data_ptr()
            if addr in seen:
                raise RuntimeError(f"SHARED PARAM MEMORY: {name} and {seen[addr]}")
            seen[addr] = name
        for name, buf in self.named_buffers():
            addr = buf.storage().data_ptr()
            if addr in seen:
                raise RuntimeError(f"SHARED BUFFER MEMORY: {name} and {seen[addr]}")
            seen[addr] = name

    def get_action_and_value(self, input_dict: dict):
        """Get action and value from the student model."""
        dist = self._actor(input_dict)
        action = dist.sample()
        value = self._critic(input_dict).flatten()
        
        logstd = self._actor.logstd
        std = torch.exp(logstd)
        neglogp = self.neglogp(action, dist.mean, std, logstd)
        
        return action, neglogp, value.flatten()
    
    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        """Get action from the student model."""
        dist = self._actor(input_dict)
        if mean:
            return dist.mean
        return dist.sample()
    
    @staticmethod
    def neglogp(x, mean, std, logstd):
        """Compute negative log probability."""
        dist = torch.distributions.Normal(mean, std)
        return -dist.log_prob(x).sum(dim=-1)


class LightweightActor(nn.Module):
    """
    Lightweight Actor with compact transformer architecture.
    """
    
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        # Always create a new tensor for logstd
        self.logstd = nn.Parameter(
            torch.ones(num_out, dtype=torch.float32) * float(config.actor_logstd),
            requires_grad=False,
        )
        self.mu = instantiate(self.config.mu_model, num_out=num_out)
        self._assert_no_shared_params_or_buffers()

    def _assert_no_shared_params_or_buffers(self):
        seen = {}
        for name, param in self.named_parameters():
            addr = param.data.storage().data_ptr()
            if addr in seen:
                raise RuntimeError(f"SHARED PARAM MEMORY in LightweightActor: {name} and {seen[addr]}")
            seen[addr] = name
        for name, buf in self.named_buffers():
            addr = buf.storage().data_ptr()
            if addr in seen:
                raise RuntimeError(f"SHARED BUFFER MEMORY in LightweightActor: {name} and {seen[addr]}")
            seen[addr] = name

    def forward(self, input_dict):
        mu = self.mu(input_dict)
        mu = torch.tanh(mu)
        std = torch.exp(self.logstd)
        dist = torch.distributions.Normal(mu, std)
        return dist


class CompactTransformer(nn.Module):
    """
    Compact Transformer for lightweight processing.
    """
    
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        
        # Input processing modules
        input_models = {}
        for input_key, input_config in config.input_models.items():
            input_models[input_key] = instantiate(input_config)
        
        self.input_models = nn.ModuleDict(input_models)
        self.feature_size = self.config.transformer_token_size * len(input_models)
        
        # Compact transformer layers
        self.sequence_pos_encoder = PositionalEncoding(config.latent_dim)
        
        # Use fewer layers and smaller dimensions for lightweight model
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=torch.nn.functional.relu,
        )
        
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=config.num_layers
        )
        
        # Output model
        if config.get("output_model", None) is not None:
            self.output_model = instantiate(config.output_model)
        else:
            self.output_model = None
    
    def forward(self, input_dict):
        # Process inputs
        processed_inputs = []
        for model_name, model in self.input_models.items():
            processed_input = model(input_dict)
            processed_inputs.append(processed_input)
        
        # Concatenate features
        features = torch.cat(processed_inputs, dim=-1)
        
        # Reshape for transformer (batch_size, seq_len, features)
        batch_size = features.shape[0]
        seq_len = features.shape[1] if len(features.shape) > 2 else 1
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
        
        # Apply positional encoding
        features = self.sequence_pos_encoder(features)
        
        # Transformer expects (seq_len, batch_size, features)
        features = features.permute(1, 0, 2)
        
        # Pass through transformer
        output = self.seqTransEncoder(features)
        
        # Take the first token output
        output = output[0]  # (batch_size, features)
        
        # Apply output model if specified
        if self.output_model is not None:
            output = self.output_model(output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).contiguous()
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :] 