import hashlib
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from scipy.io import loadmat

from cross_modal_distillation.data.available_sessions import (
    AvailableSessions,
)
from cross_modal_distillation.data.base_dataset import BaseDataset

from cross_modal_distillation.utility.preprocessing import (
    align_signal,
    common_average_reference,
    downsample,
    lowpass_filter,
)
from cross_modal_distillation.utility.utils import get_closest_indices


class FlintCODataset(BaseDataset):
    @property
    def available_sessions(self):
        return {"MonkeyC": AvailableSessions.MonkeyC.value}

    @property
    def experiment_type(self):
        return "CO"

    @property
    def download_command(self):
        self.logger.warning(
            (
                "Flint CO dataset is not available for automatic downloading. "
                "The raw dataset should be downloaded from https://crcns.org/data-sets/movements/dream/downloading-dream, "
                "by following the steps on the website."
            )
        )

    @property
    def raw_data_filenames(self):
        return [
            "Flint_2012_e1.mat",
            "Flint_2012_e2.mat",
            "Flint_2012_e3.mat",
            "Flint_2012_e4.mat",
            "Flint_2012_e5.mat",
        ]

    def _is_downloaded(self):
        raw_file_paths = [
            self.get_raw_data_file_path(filename=rfn) for rfn in self.raw_data_filenames
        ]

        for rfp in raw_file_paths:
            if not os.path.exists(rfp):
                self.logger.warning(f"Raw data file {rfp} do not exist.")
                return False
        return True

    def download_data(self):
        self.logger.error(
            (
                "Please download and transfer the data from https://crcns.org/data-sets/movements/dream/downloading-dream "
                "by following the steps on the website, and restart the training."
            )
        )
        sys.exit(1)

    def get_raw_data_file_path(self, filename):
        return os.path.join(self.raw_data_dir, filename)

    def process_raw_data(self):
        for i, filename in enumerate(self.raw_data_filenames):
            self.logger.info(
                f"Raw data processing (i.e., LFP processing) for file {filename} starts."
            )
            raw_file_path = self.get_raw_data_file_path(filename=filename)
            self.logger.info(
                f"Processing file {filename} ({i+1}/{len(self.raw_data_filenames)})..."
            )
            self.process_single_session_raw_data(file_path=raw_file_path)

    def process_single_session_raw_data(self, file_path):
        data = loadmat(file_path)
        session_day = os.path.split(file_path)[-1].split(".")[0].split("_")[-1]

        subject_data = data["Subject"]
        num_sessions = len(data["Subject"])

        for i in range(num_sessions):
            self.logger.info(
                f"Processing session ({i+1}/{num_sessions}) inside the file..."
            )
            session_data = subject_data[i][0]["Trial"]
            num_trials = len(session_data)

            lfp, t_actual = [], []
            target_pos, vel = [], []
            trial_conditions, movement_start_times, movement_end_times = [], [], []
            trial_start_times, trial_end_times = [], []
            # movement start and end correspond to center-out reach start
            # trial start and end correspond to actual trial data start and end

            t0 = session_data["Time"][0][0][0][0]
            tend = session_data["Time"][-1][0][-1][0]
            t_range = tend - t0

            num_steps = int(t_range / (self.config.delta / 1000))
            t = torch.linspace(t0, tend, num_steps)

            for j in range(num_trials):
                time_j = torch.tensor(
                    session_data["Time"][j][0].reshape(-1), dtype=torch.float32
                )
                target_pos_j = torch.tensor(
                    session_data["TargetPos"][j][0][:, :2], dtype=torch.float32
                )
                vel_j = torch.tensor(
                    session_data["HandVel"][j][0][:, :2], dtype=torch.float32
                )

                lfp_j = session_data["Neuron"][j][0]["LFP"]
                lfp_j_arr = torch.tensor(
                    np.concatenate(
                        [
                            lfp_j[k][0]
                            for k in range(len(lfp_j))
                            if lfp_j[k][0].shape[1] != 0
                        ],
                        axis=-1,
                    ),
                    dtype=torch.float32,
                )
                lfp.append(lfp_j_arr)

                t_actual.append(time_j)
                target_pos.append(target_pos_j)
                vel.append(vel_j)

                # Get the trial and movement indices
                trial_conditions.append(session_data["Condition"][j][0][0])
                trial_start_times.append(time_j[0].item())
                trial_end_times.append(time_j[-1].item())

                ## Get the movement start and end indices
                is_number = ~np.isnan(target_pos_j[:, 0])

                # Find changes in the sequence (edges of contiguous numbers)
                edges = np.diff(
                    np.r_[0, is_number, 0]
                )  # Add padding at the beginning and end

                # Find start and end indices of contiguous numbers
                movement_start_index = np.where(edges == 1)[
                    0
                ]  # Start of a number sequence
                movement_end_index = (
                    np.where(edges == -1)[0] - 1
                )  # where it reaches the target

                # even if some sessions are labelled as good, they don't have target pos
                if (
                    session_data["Condition"][j][0][0].lower() == "good"
                    and len(movement_start_index) > 0
                ):
                    for msi, mei in zip(movement_start_index, movement_end_index):
                        movement_start_times.append(time_j[msi].item())
                        movement_end_times.append(time_j[mei].item())
                else:
                    movement_start_times.append(None)
                    movement_end_times.append(None)

            lfp = torch.cat(lfp)
            t_actual = torch.cat(t_actual)
            vel = torch.cat(vel)

            # Apply CAR
            lfp = common_average_reference(lfp)

            # Downsample LFP. Raw signal fs is 2kHz and ensure alignment (sometimes lfp differs by 1 step after downsample)
            lfp = lowpass_filter(
                data=lfp,
                fs=2000,
                cutoff=(1000 / self.config.delta / 2),
            )
            lfp, _ = downsample(lfp, num_samples=t.shape[0])

            # Align vel to t
            vel = align_signal(vel, t=t_actual, align_t=t)

            # Get trial and movement start/end indices from times
            trial_start_inds = get_closest_indices(trial_start_times, t)
            trial_end_inds = get_closest_indices(trial_end_times, t)

            movement_start_inds = get_closest_indices(movement_start_times, t)
            movement_end_inds = get_closest_indices(movement_end_times, t)

            # Now, z-scoring the signals
            if self.config.z_score_kinem:
                vel_std = vel.std(dim=0)
                vel_std[vel_std == 0] = 1
                vel = (vel - vel.mean(dim=0)[None, :]) / vel_std[None, :]

            # Now, z-score LFP signal if asked (True by default)
            if "z_score_lfp" in self.config and self.config.z_score_lfp:
                data_std = lfp.std(dim=0)
                data_std[data_std == 0] = 1
                lfp = (lfp - lfp.mean(dim=0)[None, :]) / data_std[None, :]

            # Finally, save the processed raw data
            save_dict = dict(
                lfp=lfp,
                vel=vel,
                t=t,
                trial_start_inds=trial_start_inds,
                trial_end_inds=trial_end_inds,
                movement_start_inds=movement_start_inds,
                movement_end_inds=movement_end_inds,
                lfp_channel_names=[],
            )

            # Save data
            save_path = self.get_processed_raw_data_file_path(
                subject="MonkeyC", session=f"{session_day}_{i+1}"
            )
            torch.save(save_dict, save_path)

    def process_single_session_segments(self, subject, session):
        processed_raw_data_path = self.get_processed_raw_data_file_path(
            subject=subject, session=session
        )
        data = torch.load(processed_raw_data_path, weights_only=False)

        lfp = data.get("lfp", None)
        vel = data["vel"]

        lfp_channel_names = data["lfp_channel_names"]
        num_steps = lfp.shape[0]

        lfp = lfp[..., : self.config.d_lfp]

        num_steps_in_segment = int(
            self.config.segment_length / (self.config.delta / 1000)
        )
        num_discard_steps = int(num_steps % num_steps_in_segment)

        # segment the data
        segmented_lfp = rearrange(
            lfp[: (num_steps - num_discard_steps), :],
            "(b t) n -> b t n",
            t=num_steps_in_segment,
        )
        segmented_vel = rearrange(
            vel[: (num_steps - num_discard_steps), :],
            "(b t) n -> b t n",
            t=num_steps_in_segment,
        )

        # Some velocity segments are super noisy, probably monkey was not controlling the cursor at those timepoints
        ## Tukey's fences outlier detection (applied on velocity standard deviations)
        vel_stds = segmented_vel.std([1, 2])
        vel_std_threshold = (np.percentile(vel_stds, 75)) + 1.5 * (
            np.percentile(vel_stds, 75) - np.percentile(vel_stds, 25)
        )
        vel_keep_inds = vel_stds <= vel_std_threshold

        segmented_lfp = segmented_lfp[vel_keep_inds]
        segmented_vel = segmented_vel[vel_keep_inds]

        metadata_df = []
        num_segments = len(segmented_lfp)
        for i in range(num_segments):
            data_dict = dict(
                lfp=segmented_lfp[i].clone(),
                kinem=segmented_vel[i].clone(),
            )

            segment_path, segment_filename = self.save_segment_data(
                data_dict=data_dict,
                subject=subject,
                session=session,
                segment_id=i,
            )

            # set split later, comply to the metadata columns here
            meta_row = dict(
                subject=subject,
                session=session,
                subject_session=f"{subject}_{session}",
                experiment_name=self.experiment_type,
                d_lfp=lfp.shape[-1],
                d_kinem=vel.shape[-1],
                lfp_channel_names=lfp_channel_names,
                path=segment_path,
                split="train",
                segment_filename=segment_filename,
                segments_processing_str=self.segments_processing_str,
            )
            metadata_df.append(meta_row)

        # generate metadata of the current session and set splits
        metadata_df = pd.DataFrame(metadata_df)
        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

        val_size = int(self.config.val_ratio * len(metadata_df))
        test_size = int(self.config.test_ratio * len(metadata_df))

        metadata_df.loc[:val_size, "split"] = "val"
        metadata_df.loc[val_size : (val_size + test_size), "split"] = "test"

        # append the session metadata to dataset metadata
        self.metadata.concat(new_metadata_df=metadata_df)
