import hashlib
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset

from cross_modal_distillation.data.metadata import Metadata
from cross_modal_distillation.utility.utils import init_logger

torch.serialization.add_safe_globals([pd.DataFrame])
torch.serialization.add_safe_globals([dict])


class BaseDataset(Dataset):
    def __init__(self, config: DictConfig, **kwargs):
        self.config = config
        self.logger = init_logger(name=self.__class__.__name__)

        self.segments_processing_str, self.segments_processing_hash_str = (
            self.get_segments_processing_hash(
                segment_length=self.config.segment_length,
                segment_from_existing_data=self.config.segment_from_existing_data,
                existing_data_segment_length=self.config.existing_data_segment_length,
            )
        )
        self.metadata_path = self.get_metadata_path(
            segments_processing_hash_str=self.segments_processing_hash_str
        )

        # Raw data download
        os.makedirs(self.raw_data_dir, exist_ok=True)
        if not self._is_downloaded():
            self.logger.info("Raw dataset is not downloaded, download starts.")
            if self.download_data():
                self._mark_as_downloaded()
                self.logger.info("Download complete.")
            else:
                self.logger.error("Error downloading the dataset.")
                sys.exit(0)
        else:
            self.logger.info("Raw dataset is downloaded.")

        # Raw data processing
        os.makedirs(self.processed_raw_data_dir, exist_ok=True)
        if not self._is_raw_data_processed() or self.config.force_reprocess_stage1:
            self.logger.info(
                "Processed raw dataset do not exist (i.e., LFPs are not extracted) or reprocessing is enabled, processing starts."
            )
            self.process_raw_data()
            self.logger.info("Raw data processing complete.")
        else:
            self.logger.info("Processed raw data exists (i.e., LFPs are extracted.)")

        # Processing of segments from processed raw data
        os.makedirs(self.processed_segments_data_dir, exist_ok=True)

        ## Metadata initialization
        self.metadata = self.initialize_or_load_metadata()
        if not self._is_segments_processed() or self.config.force_reprocess_stage2:
            self.logger.info(
                (
                    f"Processed segments with processing string: '{self.segments_processing_str}' and hash '{self.segments_processing_hash_str}' "
                    " do not exist, or reprocessing is enabled, processing starts (paths or hashed string may have changed, check paths inside metadata)."
                )
            )

            # Empty the metadata since segments do not exist
            self.metadata.clear()

            # Process the segments now
            self.process_segments()
            self.logger.info("Processing of segments is complete.")
        else:
            self.logger.info(
                f"Processed segments exists (with processing string '{self.segments_processing_str} and hash {self.segments_processing_hash_str}')."
            )

    @property
    def column_names(self):
        return [
            "subject",
            "session",
            "subject_session",
            "experiment_name",
            "d_kinem",
            "path",
            "split",
            "segment_filename",
            "segments_processing_str",
            "d_lfp",
            "lfp_channel_names",
        ]

    @property
    def available_sessions(self):
        """
        return a list of available sessions in format of {subject}-{session}
        these will be used for
        """
        raise NotImplementedError

    @property
    def experiment_type(self):
        raise NotImplementedError

    @property
    def download_command(self):
        raise NotImplementedError

    @property
    def raw_data_dir(self):
        return os.path.join(self.config.save_dir, "raw")

    @property
    def z_score_str(self):
        if self.config.z_score_lfp:
            z_score_lfp_name = "zScLFP"
        else:
            z_score_lfp_name = "nozScLFP"

        if self.config.z_score_kinem:
            z_score_kinem_name = "zScKinem"
        else:
            z_score_kinem_name = "nozScKinem"

        return z_score_lfp_name, z_score_kinem_name

    @property
    def lfp_processing_str(self):
        lfp_processing_str = f"LFP_lpCut{self.config.lfp_lp_cutoff:.1e}Hz_hpCut{self.config.lfp_hp_cutoff:.1e}Hz"
        return lfp_processing_str

    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., LFP processing
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_raw_data_{self.config.delta:.0f}ms_{self.z_score_str[0]}_{self.z_score_str[1]}_{self.lfp_processing_str}",
        )

    @property
    def processed_segments_data_dir(self):
        """
        data dir for constructing the segmented trials from processed LFPs
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_segments_{self.segments_processing_hash_str}",
        )

    @property
    def processed_segment_pattern(self):
        return "subject_session_segid"

    @property
    def raw_data_downloaded_indicator_file_path(self):
        return os.path.join(self.raw_data_dir, "download_complete.done")

    def _is_downloaded(self):
        self.logger.debug(
            f"Raw data downloaded indicator file path: {self.raw_data_downloaded_indicator_file_path}"
        )
        return os.path.exists(self.raw_data_downloaded_indicator_file_path)

    def _mark_as_downloaded(self):
        open(self.raw_data_downloaded_indicator_file_path, "w").close()

    def _is_raw_data_processed(self):
        if not os.path.exists(self.processed_raw_data_dir):
            return False

        files_exist = []
        for subject in self.available_sessions.keys():
            for session in self.available_sessions[subject]:
                path = self.get_processed_raw_data_file_path(
                    subject=subject, session=session
                )
                files_exist.append(os.path.exists(path))
        return np.array(files_exist).all()

    def _is_segments_processed(self):
        if not os.path.exists(self.processed_segments_data_dir) or not os.path.exists(
            self.metadata_path
        ):
            return False

        if len(self.metadata):
            paths_exists = self.metadata.apply_fn_on_all_rows("path", os.path.exists)
            return paths_exists.all()
        else:
            return False

    def initialize_or_load_metadata(self) -> Metadata:
        if os.path.exists(self.metadata_path):
            metadata = Metadata(load_path=self.metadata_path)
        else:
            metadata_df = pd.DataFrame(columns=self.column_names)
            metadata = Metadata(metadata_df=metadata_df)
        return metadata

    def get_metadata_path(self, segments_processing_hash_str):
        return os.path.join(
            self.config.save_dir,
            f"metadata_{segments_processing_hash_str}.csv",
        )

    def get_raw_data_file_path(self, **kwargs):
        raise NotImplementedError

    def get_processed_raw_data_file_path(self, subject, session):
        filename = f"{subject}_{session}.pt"
        return os.path.join(self.processed_raw_data_dir, filename)

    def get_segment_path(self, **kwargs):
        if self.processed_segment_pattern == "subject_session_segid":
            segment_filename = (
                f"{kwargs['subject']}_{kwargs['session']}_{kwargs['segment_id']}.pt"
            )
            return (
                os.path.join(self.processed_segments_data_dir, segment_filename),
                segment_filename,
            )
        else:
            raise NotImplementedError(
                f"Only 'subject_session_segid' pattern is supported for saving segment files."
            )

    def get_segment_id_from_path(self, path):
        if self.processed_segment_pattern == "subject_session_segid":
            segment_filename = os.path.split(path)[-1]
            segment_id = int(segment_filename[:-3].split("_")[-1])
            return segment_id
        else:
            raise NotImplementedError(
                f"Only 'subject_session_segid' pattern is supported for saving segment files, segment_id cannot be obtained."
            )

    def download_data(self):
        success = []
        for dc in self.download_command:
            split_command = dc.split(" ")

            # Make sure the python environment that can run the download command is activated
            # in the terminal from which the code is executed,
            # otherwise, this'll throw error
            proc = subprocess.run(split_command)
            success.append(proc.returncode)
        return all([s == 0 for s in success])  # successful

    def process_single_session_raw_data(
        self, file_path, subject, session, save_data=True
    ):
        raise NotImplementedError

    def process_raw_data(self):
        """
        should call process_single_session_raw_data
        """
        raise NotImplementedError

    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
    ):
        """
        returns a tuple where the key is the processing str, value is the hashed key.
        actual str can be found in metadata.

        this part can be overwritten by each dataset class based on specific settings
        """
        processing_str = (
            f"delta{self.config.delta:.0f}ms_d_lfp{self.config.d_lfp}"
            f"_segment_length{segment_length}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
            f"_{self.z_score_str[0]}_{self.z_score_str[1]}_{self.lfp_processing_str}"
        )

        if segment_from_existing_data:
            assert (
                existing_data_segment_length is not None
            ), "Segments are asked to be created from an existing segmented data, but segment length of the existing data is None."
            processing_str += f"_from_segment_length{existing_data_segment_length}"

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str

    def process_segments(self):
        if not self.config.segment_from_existing_data:
            for subject in self.available_sessions.keys():
                sessions_count = len(self.available_sessions[subject])
                if sessions_count:
                    self.logger.info(
                        f"Segment processing for subject {subject} starts."
                    )
                    for i, session in enumerate(self.available_sessions[subject]):
                        self.logger.info(
                            f"Processing session {session} ({i+1}/{sessions_count})..."
                        )
                        self.process_single_session_segments(
                            subject=subject, session=session
                        )
        else:
            self.process_segments_from_existing_data()

        # save metadata
        self.metadata.save(self.metadata_path)

    def save_segment_data(self, data_dict, subject, session, segment_id):
        # .clone() is important while creating tensors in data_dict, as it significantly affects the size of the segment file
        # due to saving the view vs. the tensor
        segment_path, segment_filename = self.get_segment_path(
            subject=subject, session=session, segment_id=segment_id
        )
        torch.save(data_dict, segment_path)
        return segment_path, segment_filename

    def process_segments_from_existing_data(self):
        _, existing_segments_processing_hash_str = self.get_segments_processing_hash(
            segment_length=self.config.existing_data_segment_length,
            segment_from_existing_data=False,
        )

        existing_metadata_path = self.get_metadata_path(
            segments_processing_hash_str=existing_segments_processing_hash_str
        )
        existing_metadata = Metadata(load_path=existing_metadata_path)

        self.logger.info(
            f"Processing {self.config.segment_length} second segments from {self.config.existing_data_segment_length} second segments with metadata_{existing_segments_processing_hash_str}.csv..."
        )

        new_metadata_df = []

        global_new_segment_id = 0
        for row_id in range(len(existing_metadata)):
            row = existing_metadata.get_row_by_index(row_id)
            existing_segment_data = torch.load(row.path)
            segment_id = self.get_segment_id_from_path(path=row.path)

            # Create the new LFP and kinematic segments
            new_segment_data = {}
            for name in existing_segment_data.keys():
                existing_num_steps = existing_segment_data[name].shape[0]
                new_num_steps_in_segment = int(
                    self.config.segment_length / (self.config.delta / 1000)
                )
                num_discard_steps = int(existing_num_steps % new_num_steps_in_segment)
                new_data = rearrange(
                    existing_segment_data[name][
                        : (existing_num_steps - num_discard_steps), ...
                    ],
                    "(b t) n -> b t n",
                    t=new_num_steps_in_segment,
                )

                if num_discard_steps > 0:
                    new_data = list(new_data)
                    new_data.append(
                        existing_segment_data[name][
                            (existing_num_steps - num_discard_steps) :
                        ]
                    )
                new_segment_data[name] = new_data
                num_new_segments = len(new_data)

            # save the new segment and create the new metadata
            for split_id in range(num_new_segments):
                new_segment_id = segment_id * num_new_segments + split_id

                data_dict = {
                    k: v[split_id].clone() for k, v in new_segment_data.items()
                }
                new_segment_path, new_segment_filename = self.save_segment_data(
                    data_dict=data_dict,
                    subject=row.subject,
                    session=row.session,
                    segment_id=new_segment_id,
                )
                global_new_segment_id += 1

                new_meta_row = row.to_dict()
                new_meta_row["path"] = new_segment_path
                new_meta_row["segment_filename"] = new_segment_filename
                new_meta_row["segments_processing_str"] = self.segments_processing_str
                new_metadata_df.append(new_meta_row)

        new_metadata_df = pd.DataFrame(new_metadata_df)
        self.metadata.concat(new_metadata_df=new_metadata_df)

    def process_single_session_segments(self, subject, session):
        raise NotImplementedError
