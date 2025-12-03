import os
from typing import List, Optional, Union

import sys
import torch
import torch.nn as nn
from cross_modal_distillation.build import build_module
from cross_modal_distillation.data.metadata import Metadata
from cross_modal_distillation.utility.utils import init_logger, to_cuda, to_dtype
from tqdm import tqdm
from einops import repeat

std_logger = init_logger("Model")


class Model(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        ckpt_metadata_path: str,
    ):
        """
        Initializing the model
        To load the model correctly, we need 1) the model weights and 2) the metadata of the dataset
        used to train the model.
        """
        super().__init__()

        self.ckpt_path = ckpt_path
        self.ckpt_metadata_path = ckpt_metadata_path
        self.load_checkpoint()

    def load_checkpoint(self):
        """
        Loads the model components: tokenizer and backbone
        """
        if not os.path.exists(self.ckpt_path):
            std_logger.error(f"Checkpoint path does not exist: {self.ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        metadata = Metadata(load_path=self.ckpt_metadata_path)
        config = ckpt["config"]

        # Initialize tokenizer module
        self.tokenizer = build_module(
            config=config.tokenizer,
            session_d_input_dict=metadata.get_subject_session_d_input(),
        )

        # Initialize feature extractor backbone
        self.backbone = build_module(config=config.backbone)
        std_logger.info("Model components initialized successfully")

        # Load the parameters
        self.load_state_dict(ckpt["state_dict"])
        std_logger.info("Model components loaded successfully")

    def pool_embeddings(
        self,
        embeddings,
        position_ids,
        seq_lens=None,
    ):
        B, N, D = embeddings.shape

        if seq_lens is not None and B == 1:
            # means we are operating with variable length
            pooled_embeddings = []
            pooled_position_ids = []
            pooled_seq_lens = []

            embeddings_split = torch.split(embeddings, seq_lens, dim=1)
            position_ids_split = torch.split(position_ids, seq_lens, dim=1)
            for embeddings_split_this, position_ids_split_this in zip(
                embeddings_split, position_ids_split
            ):
                # assumes all timesteps exist
                max_position = position_ids_split_this.max()

                if len(position_ids_split_this.shape) == 3:
                    if position_ids_split_this.shape[-1] == 1:
                        position_ids_split_this = position_ids_split_this.squeeze(
                            dim=-1
                        )
                    else:
                        raise ValueError(
                            f"'position_ids' must be a 2D tensor or a 3D tensor with last dimension being 1, but received: {position_ids.shape}."
                        )
                position_ids_split_this = repeat(
                    position_ids_split_this,
                    "b n -> b n d",
                    d=D,
                ).long()

                pooled_embeddings_this = torch.zeros(1, max_position + 1, D).to(
                    embeddings_split_this
                )
                pooled_embeddings_this = pooled_embeddings_this.scatter_reduce(
                    1,
                    position_ids_split_this,
                    embeddings_split_this,
                    reduce="mean",
                )

                pooled_embeddings.append(pooled_embeddings_this)
                pooled_position_ids.append(torch.arange(max_position + 1))
                pooled_seq_lens.append(max_position + 1)
        else:
            max_position = position_ids.max()
            position_ids = position_ids.long()

            if len(position_ids.shape) == 3:
                if position_ids.shape[-1] == 1:
                    position_ids = position_ids.squeeze(dim=-1)
                else:
                    raise ValueError(
                        f"'position_ids' must be a 2D tensor or a 3D tensor with last dimension being 1, but received: {position_ids.shape}."
                    )
            position_ids = repeat(position_ids, "b n -> b n d", d=D)

            pooled_embeddings = torch.zeros(B, max_position + 1, D).to(embeddings)
            pooled_embeddings = pooled_embeddings.scatter_reduce(
                1, position_ids.long(), embeddings, reduce="mean"
            )
            pooled_position_ids = repeat(
                torch.arange(max_position + 1), "n -> b n", b=B
            )
            pooled_seq_lens = [max_position + 1] * B
        return pooled_embeddings, pooled_position_ids, pooled_seq_lens

    def forward(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        subject_sessions: List[str],
        position_ids: Optional[Union[torch.Tensor, List]] = None,
        **kwargs,
    ):
        _, tokens, position_ids, _, seq_lens = self.tokenizer(
            x=inputs,
            position_ids=position_ids,
            subject_sessions=subject_sessions,
        )
        _, embeddings = self.backbone(
            x=tokens, position_ids=position_ids, seq_lens=seq_lens
        )
        pooled_embeddings, pooled_position_ids, pooled_seq_lens = self.pool_embeddings(
            embeddings=embeddings,
            position_ids=position_ids,
            seq_lens=seq_lens,
        )

        return pooled_embeddings

    @torch.no_grad
    def get_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_embeddings: Optional[bool] = False,
        save_dir: Optional[str] = ".",
        device: Optional[str] = "cuda:0",
        return_embeddings: Optional[bool] = False,
    ):
        self.eval()
        self.tokenizer = self.tokenizer.to(device)
        self.backbone = self.backbone.to(device)

        pooled_embeddings_all = []
        segment_filenames_all = []
        with tqdm(dataloader, unit="batch") as tbatch:
            std_logger.info("Embedding extraction starts.")
            for batch_idx, (
                inputs,
                subject_sessions,
                position_ids,
                segment_filenames,
            ) in enumerate(tbatch):
                tbatch.set_description(f"Batch {batch_idx + 1}/{len(dataloader)}")

                with torch.amp.autocast(device):
                    inputs = to_cuda(inputs, device=device)
                    position_ids = to_cuda(position_ids, device=device)
                    pooled_embeddings = self.forward(
                        inputs=inputs,
                        subject_sessions=subject_sessions,
                        position_ids=position_ids,
                    )
                pooled_embeddings = to_dtype(pooled_embeddings, dtype="float32")
                pooled_embeddings_all.extend(list(pooled_embeddings))
                segment_filenames_all.extend(segment_filenames)

        if save_embeddings:
            os.makedirs(save_dir, exist_ok=True)
            std_logger.info(f"Saving extracted embeddings at {save_dir}")
            for pooled_embedding, segment_fname in zip(
                pooled_embeddings_all, segment_filenames_all
            ):
                torch.save(
                    pooled_embedding.float().cpu(), f"{save_dir}/{segment_fname}"
                )
        if return_embeddings:
            return pooled_embeddings_all, segment_filenames_all
        return
