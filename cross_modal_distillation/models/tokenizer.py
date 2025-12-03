from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from cross_modal_distillation.models.causal_conv1d import CausalConv1d
from cross_modal_distillation.models.MLP import MLP
from cross_modal_distillation.utility.utils import init_logger

std_logger = init_logger("Tokenizer")


class PatchTokenizer(nn.Module):
    def __init__(
        self,
        spatial_patch_size: int,
        session_d_input_dict: Dict,
        d_hidden: int,
        layer_list: List = [512],
        activation: str = "tanh",
        dropout: float = 0.1,
        learn_patch_embedding: bool = False,
        use_embedding_for_input: bool = False,
        use_conv_for_input: bool = False,
        kernel_size: int = 3,
        dilation: int = 1,
        max_count: int = 5,
        initialization_std: float = 1,
        **kwargs,
    ):
        """
        as this module is not as complicated as model.py or tasks.py in which modules can have configurable module
        attributes with complex configs, we pass config parameters explicitly rather than under config field
        """
        super().__init__()

        self.spatial_patch_size = spatial_patch_size
        self.d_hidden = d_hidden
        self.layer_list = layer_list
        self.activation = activation
        self.dropout = dropout

        self.learn_patch_embedding = learn_patch_embedding
        self.use_embedding_for_input = use_embedding_for_input
        self.use_conv_for_input = use_conv_for_input
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.max_count = max_count
        self.session_d_input_dict = session_d_input_dict
        self.initialization_std = initialization_std

        if self.use_embedding_for_input:
            assert (
                self.d_hidden % self.spatial_patch_size == 0
            ), f"'d_hidden' ({self.d_hidden}) must be divisible by 'spatial_patch_size' ({self.spatial_patch_size}) for PatchTokenizer."
            embedding_size = self.d_hidden // self.spatial_patch_size

            self.pad_value = self.max_count + 1
            self.embedder = nn.Embedding(
                self.max_count + 2, embedding_size, padding_idx=self.pad_value
            )
        elif self.use_conv_for_input:
            self.embedder = CausalConv1d(
                in_channels=self.spatial_patch_size,
                out_channels=self.d_hidden,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
            )
            self.pad_value = 0
        else:
            self.embedder = MLP(
                d_input=self.spatial_patch_size,
                d_out=self.d_hidden,
                layer_list=self.layer_list,
                activation=self.activation,
                dropout=self.dropout,
            )
            self.pad_value = 0
        self.embedder.apply(self._init_weights)

        # Create session-specific patch embeddings
        self.set_patch_embeddings()
        if self.patch_embeddings:
            self.patch_embeddings.apply(self._init_weights)

    def __deepcopy__(self, memo):
        copy_module = PatchTokenizer(
            spatial_patch_size=self.spatial_patch_size,
            session_d_input_dict=self.session_d_input_dict,
            d_hidden=self.d_hidden,
            layer_list=self.layer_list,
            activation=self.activation,
            dropout=self.dropout,
            learn_patch_embedding=self.learn_patch_embedding,
            use_embedding_for_input=self.use_embedding_for_input,
            use_conv_for_input=self.use_conv_for_input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            max_count=self.max_count,
            initialization_std=self.initialization_std,
        )
        copy_module.load_state_dict(self.state_dict())
        return copy_module

    def _init_weights(
        self,
        module,
    ):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.initialization_std)

    def get_num_spatial_patches(self, d_input):
        num_patches = (
            d_input // self.spatial_patch_size + 1
            if d_input % self.spatial_patch_size > 0
            else d_input // self.spatial_patch_size
        )
        return num_patches

    def set_patch_embeddings(self):
        if self.learn_patch_embedding:
            self.patch_embeddings = nn.ModuleDict()
            for ss, d_input in self.session_d_input_dict.items():
                num_patches = self.get_num_spatial_patches(d_input)
                self.patch_embeddings[ss] = nn.Embedding(num_patches, self.d_hidden)
        else:
            self.patch_embeddings = None

    def update_for_new_sessions(
        self,
        new_session_d_input_dict: Dict[str, int],
        **kwargs,
    ):
        self.session_d_input_dict.update(new_session_d_input_dict)

        new_modules = []

        if self.patch_embeddings is not None:
            for ss, d_input in new_session_d_input_dict.items():
                if ss not in self.patch_embeddings:
                    num_patches = self.get_num_spatial_patches(d_input)
                    self.patch_embeddings[ss] = nn.Embedding(num_patches, self.d_hidden)
                    std_logger.warning(
                        f"Adding module 'patch_embeddings.{ss}' to tokenizer..."
                    )
                    new_modules.append(f"patch_embeddings.{ss}")
        return new_modules

    def clamp_tensor(self, x):
        if self.use_embedding_for_input:
            x = x.long().clamp(min=0, max=self.max_count)
        return x

    def _create_patches_for_batched_tensor(
        self, x: torch.Tensor, position_ids: Optional[Union[torch.Tensor, List]] = None
    ):
        assert x.dim() == 3

        B, N, D = x.shape

        spatial_pad_size = (
            (self.spatial_patch_size - D % self.spatial_patch_size)
            if D % self.spatial_patch_size != 0
            else 0
        )

        x = self.clamp_tensor(x)

        x_patched = torch.nn.functional.pad(
            x, (0, spatial_pad_size, 0, 0), value=self.pad_value
        )

        assert x_patched.shape[0] == B
        assert x_patched.shape[1] == N
        assert x_patched.shape[2] % self.spatial_patch_size == 0

        x_patched = rearrange(
            x_patched,
            "b n (nsp sps) -> b n nsp sps",
            sps=self.spatial_patch_size,
        )

        if position_ids is None:
            position_ids = torch.arange(x_patched.shape[1]).to(x_patched.device)
            position_ids = repeat(
                position_ids,
                "n -> b n",
                b=B,
            )
        position_ids_patched = repeat(
            position_ids, "b n -> b (n nsp)", nsp=x_patched.shape[2]
        )

        patch_ids = torch.arange(x_patched.shape[2]).to(x_patched.device)
        patch_ids = repeat(
            patch_ids,
            "nsp -> b (n nsp)",
            b=B,
            n=x_patched.shape[1],
        )

        x_patched = rearrange(x_patched, "b n nsp sps -> b (n nsp) sps")
        seq_lens_patched = [x_patched.shape[1]] * x_patched.shape[0]

        assert x_patched.dim() == 3
        assert x_patched.shape[0] == B
        assert x_patched.shape[2] == self.spatial_patch_size

        return x_patched, position_ids_patched, patch_ids, seq_lens_patched

    def create_patches(
        self,
        x: Union[torch.Tensor, List],
        position_ids: Optional[Union[torch.Tensor, List]] = None,
    ):
        if isinstance(x, torch.Tensor):
            x_patched, position_ids_patched, patch_ids, seq_lens_patched = (
                self._create_patches_for_batched_tensor(x, position_ids=position_ids)
            )
        else:
            # i.e. we have a list of N*D_i tensors
            # need to process them one by one
            # used for supporting variable input lengths sequence

            x_patched, position_ids_patched, patch_ids, seq_lens_patched = (
                [],
                [],
                [],
                [],
            )

            for i, x_this in enumerate(x):
                if position_ids is not None:
                    position_ids_this = position_ids[i]
                else:
                    position_ids_this = None

                (
                    x_patched_batched,
                    position_ids_patched_batched,
                    patch_ids_batched,
                    seq_lens_patched_batched,
                ) = self._create_patches_for_batched_tensor(
                    x_this.unsqueeze(0), position_ids=position_ids_this.unsqueeze(0)
                )

                x_patched.append(x_patched_batched[0])
                position_ids_patched.append(position_ids_patched_batched[0])
                patch_ids.append(patch_ids_batched[0])
                seq_lens_patched.append(seq_lens_patched_batched[0])

            x_patched = torch.cat(x_patched).unsqueeze(dim=0)
            position_ids_patched = torch.cat(position_ids_patched).unsqueeze(dim=0)
            patch_ids = torch.cat(patch_ids).unsqueeze(dim=0)
        return x_patched, position_ids_patched, patch_ids, seq_lens_patched

    def add_patch_embeddings(
        self, x, x_patched, patch_ids, seq_lens_patched, subject_sessions
    ):
        if self.patch_embeddings:
            patch_embeddings = []
            if isinstance(x, list):
                patch_ids_split = torch.split(patch_ids, seq_lens_patched, dim=1)
            else:
                patch_ids_split = patch_ids

            for i, ss in enumerate(subject_sessions):
                patch_embedding_i = self.patch_embeddings[ss](patch_ids_split[i])
                patch_embeddings.append(patch_embedding_i)

            if isinstance(x, list):
                patch_embeddings = torch.cat(patch_embeddings, dim=1)
            else:
                patch_embeddings = torch.stack(patch_embeddings, dim=0)
            x_patched += patch_embeddings
        return x_patched

    def forward(
        self,
        x: Union[torch.Tensor, List],
        subject_sessions: List,
        position_ids: Optional[Union[torch.Tensor, List]] = None,
        **kwargs,
    ):
        """
        x: Input tensor of shape (B, N, D) or a list of tensors each of shape (N, D_i)
            B: Batch size
            N: Time points
            D: Channel dim

            If N and D is not divisible by spatial and temporal patch size, input tensor will be padded.
        """
        x_patched, position_ids_patched, patch_ids, seq_lens_patched = (
            self.create_patches(x=x, position_ids=position_ids)
        )  # b n sps
        data_patched = x_patched

        # Important when we want to test another session on a model trained on one session,
        # TODO: For now, this is only supported on models trained on one session
        if len(self.patch_embeddings) == 1:
            subject_sessions = list(self.patch_embeddings.keys()) * len(
                subject_sessions
            )

        if self.use_embedding_for_input:
            x_patched = self.embedder(x_patched)
            x_patched = rearrange(x_patched, "b n ... d e -> b n ... (d e)")
        else:
            x_patched = self.embedder(x_patched)

        # add session-specific patch embeddings
        x_patched = self.add_patch_embeddings(
            x=x,
            x_patched=x_patched,
            patch_ids=patch_ids,
            seq_lens_patched=seq_lens_patched,
            subject_sessions=subject_sessions,
        )

        return (
            data_patched,
            x_patched,
            position_ids_patched,
            patch_ids,
            seq_lens_patched,
        )
