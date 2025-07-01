import torch
import torch.nn as nn
import numpy as np
import math

from models.modules import (
    ParityBackbone, SynapseUNET, Squeeze, SuperLinear,
    LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding,
    CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
)
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class ContinuousThoughtMachine(nn.Module):
    """
    Continuous Thought Machine (CTM).

    Technical report: https://arxiv.org/abs/2505.05522
    Interactive Website: https://pub.sakana.ai/ctm/
    Blog: https://sakana.ai/ctm/
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',
                 n_random_pairing_self=0):
        super().__init__()

        # Store arguments
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.positional_embedding_type = positional_embedding_type
        self.out_dims = out_dims
        self.neuron_select_type = neuron_select_type
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        # Validate arguments
        self.verify_args()

        # Determine backbone output dimension
        d_backbone = self.get_d_backbone()

        # Input / backbone setup
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)

        # Static Linear projections (no LazyLinear)
        if heads:
            # KV proj: d_backbone → d_input
            self.kv_proj = nn.Sequential(
                nn.Linear(d_backbone, d_input),
                nn.LayerNorm(d_input)
            )
            # Q proj: n_synch_action → d_input
            self.q_proj = nn.Linear(n_synch_action, d_input)
            self.attention = nn.MultiheadAttention(
                embed_dim=d_input,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.kv_proj = None
            self.q_proj = None
            self.attention = None

        # Core CTM modules
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(
            deep_nlms,
            do_layernorm_nlm,
            memory_length,
            memory_hidden_dims,
            d_model,
            dropout_nlm
        )

        # Recurrent start states
        self.register_parameter(
            'start_activated_state',
            nn.Parameter(
                torch.zeros(d_model)
                     .uniform_(-math.sqrt(1/d_model),
                              math.sqrt(1/d_model))
            )
        )
        self.register_parameter(
            'start_trace',
            nn.Parameter(
                torch.zeros(d_model, memory_length)
                     .uniform_(-math.sqrt(1/(d_model+memory_length)),
                              math.sqrt(1/(d_model+memory_length)))
            )
        )

        # Synchronisation setup
        self.neuron_select_type_out, self.neuron_select_type_action = \
            self.get_neuron_select_type()
        self.synch_representation_size_action = \
            self.calculate_synch_representation_size(n_synch_action)
        self.synch_representation_size_out = \
            self.calculate_synch_representation_size(n_synch_out)

        for synch_type, size in (
            ('action', self.synch_representation_size_action),
            ('out',   self.synch_representation_size_out)
        ):
            print(f"Synch representation size {synch_type}: {size}")

        if self.synch_representation_size_action:
            self.set_synchronisation_parameters(
                'action', n_synch_action, n_random_pairing_self
            )
        self.set_synchronisation_parameters(
            'out', n_synch_out, n_random_pairing_self
        )

        # Output projector (static Linear)
        self.output_projector = nn.Linear(
            self.synch_representation_size_out,
            self.out_dims
        )

    def get_d_backbone(self):
        """
        Return the output channel dimension of the backbone.
        'none' → 12 raw board planes.
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            base, scale = self.backbone_type.split('-')
            scale = int(scale)
            mapping = {
                '18': [64,128,256,512],
                '34': [64,128,256,512],
                '50': [256,512,1024,2048],
                '101':[256,512,1024,2048],
                '152':[256,512,1024,2048]
            }
            return mapping[base.replace('resnet','')][scale-1]
        elif self.backbone_type == 'none':
            return 12
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_initial_rgb(self):
        """
        Adapt input channels for ResNet if needed.
        """
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3,1,1)
        else:
            self.initial_rgb = nn.Identity()

    def set_backbone(self):
        """
        Instantiate backbone module.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2,
                                           d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        Instantiate positional embedding.
        """
        t = self.positional_embedding_type
        if t == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone,
                                                      gamma=1/2.5)
        elif t == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif t == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif t == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif t == 'none':
            return lambda x: 0
        else:
            raise ValueError(f"Invalid positional_embedding_type: {t}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm,
                                memory_length, memory_hidden_dims,
                                d_model, dropout):
        """
        Build neuron-level models (SuperLinear+GLU or linear).
        """
        if deep_nlms:
            return nn.Sequential(
                SuperLinear(memory_length,
                            out_dims=2*memory_hidden_dims,
                            N=d_model,
                            do_norm=do_layernorm_nlm,
                            dropout=dropout),
                nn.GLU(),
                SuperLinear(memory_hidden_dims,
                            out_dims=2, N=d_model,
                            do_norm=do_layernorm_nlm,
                            dropout=dropout),
                nn.GLU(),
                Squeeze(-1)
            )
        else:
            return nn.Sequential(
                SuperLinear(memory_length,
                            out_dims=2, N=d_model,
                            do_norm=do_layernorm_nlm,
                            dropout=dropout),
                nn.GLU(),
                Squeeze(-1)
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        """
        Build synapse module (1-layer or UNet).
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model*2, d_model*2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def calculate_synch_representation_size(self, n_synch):
        """
        Compute size of sync representation.
        """
        t = self.neuron_select_type
        if t == 'random-pairing':
            return n_synch
        elif t in ('first-last','random'):
            return (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron_selection_type: {t}")

    def get_neuron_select_type(self):
        """
        Map selection string to side-types.
        """
        print(f"Using neuron select type: {self.neuron_select_type}")
        t = self.neuron_select_type
        if t == 'first-last':
            return 'first', 'last'
        elif t in ('random','random-pairing'):
            return t, t
        else:
            raise ValueError(f"Invalid neuron_selection_type: {t}")

    def set_synchronisation_parameters(self, synch_type, n_synch,
                                       n_random_pairing_self=0):
        """
        Register neuron index buffers and decay parameters.
        """
        left, right = self.initialize_left_right_neurons(
            synch_type,
            self.d_model,
            n_synch,
            n_random_pairing_self
        )
        rep_size = (self.synch_representation_size_action
                    if synch_type=='action'
                    else self.synch_representation_size_out)
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(
            f'decay_params_{synch_type}',
            nn.Parameter(torch.zeros(rep_size), requires_grad=True)
        )

    def initialize_left_right_neurons(self, synch_type, d_model,
                                      n_synch,
                                      n_random_pairing_self=0):
        """
        Generate neuron index buffers for sync.
        """
        t = self.neuron_select_type
        if t == 'first-last':
            if synch_type == 'out':
                idxs = torch.arange(0, n_synch)
            else:
                idxs = torch.arange(d_model-n_synch, d_model)
            return idxs, idxs
        elif t == 'random':
            left  = torch.from_numpy(np.random.choice(d_model, n_synch))
            right = torch.from_numpy(np.random.choice(d_model, n_synch))
        elif t == 'random-pairing':
            left  = torch.from_numpy(np.random.choice(d_model, n_synch))
            right = torch.cat([
                left[:n_random_pairing_self],
                torch.from_numpy(np.random.choice(d_model, n_synch-n_random_pairing_self))
            ])
        device = self.start_activated_state.device
        return left.to(device), right.to(device)

    def verify_args(self):
        """
        Check validity of constructor arguments.
        """
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron_select_type: {self.neuron_select_type}"
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("No positional embedding if backbone_type is none.")

    def compute_synchronisation(self, activated_state,
                                decay_alpha, decay_beta,
                                r, synch_type):
        """
        Compute neuron synchronization vector.
        """
        if synch_type == 'action':
            n_synch = self.n_synch_action
            left    = self.action_neuron_indices_left
            right   = self.action_neuron_indices_right
        else:
            n_synch = self.n_synch_out
            left    = self.out_neuron_indices_left
            right   = self.out_neuron_indices_right

        t = self.neuron_select_type
        if t in ('first-last', 'random'):
            if t == 'first-last':
                if synch_type == 'action':
                    sel = activated_state[:, -n_synch:]
                else:
                    sel = activated_state[:, :n_synch]
                selected_left  = selected_right = sel
            else:
                selected_left  = activated_state[:, left]
                selected_right = activated_state[:, right]
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise = outer[:, i, j]
        else:  # random-pairing
            leftv  = activated_state[:, left]
            rightv = activated_state[:, right]
            pairwise = leftv * rightv

        if decay_alpha is None:
            decay_alpha = pairwise
            decay_beta  = torch.ones_like(pairwise)
        else:
            decay_alpha = r * decay_alpha + pairwise
            decay_beta  = r * decay_beta  + 1

        synchronization = decay_alpha / torch.sqrt(decay_beta)
        return synchronization, decay_alpha, decay_beta

    def compute_features(self, x):
        """
        Compute KV features for attention from raw input.
        """
        init = self.initial_rgb(x)
        feats = self.backbone(init)
        pos   = self.positional_embedding(feats)
        combined = (feats + pos).flatten(2).transpose(1,2)
        return self.kv_proj(combined)

    def compute_certainty(self, current_prediction):
        """
        Compute 1 - normalized entropy, stacked as [ne, 1-ne].
        """
        B = current_prediction.size(0)
        resh = current_prediction.reshape([B] + self.prediction_reshaper)
        ne   = compute_normalized_entropy(resh)
        return torch.stack((ne, 1-ne), -1)

    def forward(self, x, track=False):
        """
        Forward pass over T internal iterations:
          - compute synchronization_action
          - multihead attention → synapses → NLMs
          - compute synchronization_out → prediction & certainty
        """
        B = x.size(0)
        device = x.device
        kv = self.compute_features(x)

        state_trace     = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device)
        certainties = torch.empty(B, 2, self.iterations, device=device)

        decay_alpha_action = decay_beta_action = None
        decay_alpha_out    = decay_beta_out    = None
        # clamp decay params
        self.decay_params_action.data.clamp_(0, 15)
        self.decay_params_out.data.clamp_(0, 15)
        r_action = torch.exp(-self.decay_params_action
                             ).unsqueeze(0).repeat(B, 1)
        r_out    = torch.exp(-self.decay_params_out
                             ).unsqueeze(0).repeat(B, 1)

        for stepi in range(self.iterations):
            # Synchronization for action
            sync_action, decay_alpha_action, decay_beta_action = \
                self.compute_synchronisation(
                    activated_state,
                    decay_alpha_action,
                    decay_beta_action,
                    r_action,
                    'action'
                )

            # Attention
            q = self.q_proj(sync_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(
                q, kv, kv,
                average_attn_weights=False,
                need_weights=True
            )
            attn_out = attn_out.squeeze(1)
            pre = torch.cat((attn_out, activated_state), dim=-1)

            # Synapses + trace update
            state = self.synapses(pre)
            state_trace = torch.cat(
                (state_trace[:,:,1:], state.unsqueeze(-1)),
                dim=-1
            )

            # Neuron-Level Models
            activated_state = self.trace_processor(state_trace)

            # Synchronization for output
            sync_out, decay_alpha_out, decay_beta_out = \
                self.compute_synchronisation(
                    activated_state,
                    decay_alpha_out,
                    decay_beta_out,
                    r_out,
                    'out'
                )

            # Prediction & Certainty
            pred = self.output_projector(sync_out)
            cert = self.compute_certainty(pred)

            predictions[..., stepi] = pred
            certainties[..., stepi] = cert

        if track:
            # optionally return tracked internals
            # user can extend as needed
            return predictions, certainties

        return predictions, certainties, sync_out