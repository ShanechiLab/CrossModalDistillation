import os

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from scipy import signal

from cross_modal_distillation.data.available_sessions import AvailableSessions
from cross_modal_distillation.data.base_dataset import BaseDataset
from cross_modal_distillation.utility.preprocessing import (
    align_signal,
    common_average_reference,
    downsample,
    downsample_in_chunks,
    highpass_filter,
    lowpass_filter,
    notch_filter,
)
from cross_modal_distillation.utility.utils import int2str


class MakinRTDataset(BaseDataset):
    @property
    def available_sessions(self):
        """
        return a list of available sessions in format of {subject}-{session}
        these will be used for
        """
        return {"MonkeyI": AvailableSessions.MonkeyI.value}

    @property
    def experiment_type(self):
        return "RT"

    @property
    def download_command(self):
        return [
            f"zenodo_get 3854034 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1488440 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1486147 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1484824 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1473703 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1467953 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1467050 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1451793 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1433942 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1432818 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1421880 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1421310 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1419774 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1419172 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1413592 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1412635 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1412094 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1411978 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1411882 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1411474 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1410423 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1321264 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1321256 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1303720 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1302866 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1302832 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1301045 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1167965 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1163026 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.1161225 --output-dir {self.raw_data_dir}",
            f"zenodo_get --doi 10.5281/zenodo.854733 --output-dir {self.raw_data_dir}",
        ]

    @property
    def subject_name_map(self):
        return {"MonkeyI": "indy"}

    def get_raw_data_file_path(self, subject, session):
        filename = f"{self.subject_name_map[subject]}_{session}"
        return [
            os.path.join(self.raw_data_dir, f"{filename}.mat"),
            os.path.join(self.raw_data_dir, f"{filename}.nwb"),
        ]

    def process_raw_data(self):
        for subject in self.available_sessions.keys():
            self.logger.info(
                f"Raw data processing (i.e., LFP processing) for subject {subject} starts."
            )

            sessions_count = len(self.available_sessions[subject])
            for i, session in enumerate(self.available_sessions[subject]):
                raw_file_path = self.get_raw_data_file_path(
                    subject=subject, session=session
                )
                self.logger.info(
                    f"Processing session {session} ({i+1}/{sessions_count})..."
                )

                self.process_single_session_raw_data(
                    file_path=raw_file_path, subject=subject, session=session
                )

    def process_single_session_raw_data(
        self, file_path, subject, session, save_data=True
    ):

        kinem_file_path, lfp_file_path = file_path
        with h5py.File(kinem_file_path, "r") as f:
            t = f["t"][0]

            t_range = t[-1] - t[0]
            num_steps = int(t_range / (self.config.delta / 1000))

            # get cursor velocity
            pos = f["cursor_pos"][:].T
            vel = np.gradient(pos, t, axis=0)

            # get trial start and trial and indices
            target_pos = f["target_pos"][:].T

        # resampling of cursor position and velocity
        vel, t = signal.resample(vel, num_steps, t=t)

        # discard the first trial, cursor movement do not start as soon as trial starts
        trial_start_inds = np.diff(target_pos, axis=0).sum(axis=1) != 0

        # also discard the last one, it'll be a short one probably
        trial_start_inds = np.where(trial_start_inds)[0] + 1

        # and downsample indices to desired delta
        trial_start_inds = (
            (trial_start_inds) // (target_pos.shape[0] / num_steps)
        ).astype("int32")
        first_trial_ind, last_trial_ind = trial_start_inds[0], trial_start_inds[-1]

        # remove until the first trial, and after the last
        vel = torch.tensor(vel[first_trial_ind:last_trial_ind, :], dtype=torch.float32)
        t = torch.tensor(t[first_trial_ind:last_trial_ind], dtype=torch.float32)
        trial_start_inds = torch.tensor(
            trial_start_inds[:-1] - first_trial_ind, dtype=torch.int32
        )

        if self.config.z_score_kinem:
            vel_std = vel.std(dim=0)
            vel_std[vel_std == 0] = 1
            vel = (vel - vel.mean(dim=0)[None, :]) / vel_std[None, :]

        # Process LFPs
        # Now, load the LFP broadband signals
        nwb_file = h5py.File(lfp_file_path)
        conversion = nwb_file["acquisition"]["timeseries"]["broadband"]["data"].attrs[
            "conversion"
        ]
        broadband_signal = (
            np.array(nwb_file["acquisition"]["timeseries"]["broadband"]["data"])
            * conversion
        )
        broadband_t = np.array(
            nwb_file["acquisition"]["timeseries"]["broadband"]["timestamps"]
        )
        lfp_channel_names = np.array(
            nwb_file["acquisition"]["timeseries"]["broadband"][
                "electrode_names"
            ].asstr()[()],
            dtype="str",
        ).tolist()

        # Broadband signal sampling frequency, it'll be 24414.3378 Hz but reported as 24414.0625Hz
        broadband_fs = 1 / (broadband_t[1] - broadband_t[0])
        intermediary_fs = broadband_fs / 32  # ~763Hz
        target_fs = 1000 / self.config.delta

        ## 1. Anti-aliasing before initial downsampling (optional due to speed, empirical results did not change when False)
        broadband_signal = lowpass_filter(
            data=broadband_signal,
            fs=broadband_fs,
            cutoff=(intermediary_fs / 2),
        )

        ## 2. Downsample signal to 763 Hz
        raw_signal, broadband_ds_t = downsample_in_chunks(
            data=broadband_signal,
            t=broadband_t,
            original_fs=broadband_fs,
            target_fs=intermediary_fs,
        )
        raw_signal -= raw_signal.mean(axis=0)

        ## 3. Notch filtering 60Hz line noise and its harmonics, 60Hz in US, 50Hz in Europe
        raw_signal = notch_filter(
            data=raw_signal, fs=intermediary_fs, freq=60, quality_factor=30
        )

        ## 4. Highpass filter to remove low frequencies such as DC part
        raw_signal = highpass_filter(
            data=raw_signal,
            fs=intermediary_fs,
            cutoff=self.config.lfp_hp_cutoff,  # e.g., 0.05 to remove DC
        )

        ## 5. Lowpass filter to remove high frequencies (anti-aliasing)
        raw_signal = lowpass_filter(
            data=raw_signal,
            fs=intermediary_fs,
            cutoff=self.config.lfp_lp_cutoff,  # e.g., 50 Hz as we downsample to 100 Hz at the end
        )

        ## 6. Common-average-referencing
        raw_signal = common_average_reference(raw_signal)

        ## 7. Downsampling to target frequency
        raw_signal, broadband_ds_t = downsample(
            data=raw_signal,
            t=broadband_ds_t,
            original_fs=intermediary_fs,
            target_fs=target_fs,
        )

        ## 8. Align LFP to spiking signal
        lfp = align_signal(data=raw_signal, t=broadband_ds_t, align_t=t)

        # Now, z-score LFP signal if asked (True by default)
        if self.config.z_score_lfp:
            data_std = lfp.std(dim=0)
            data_std[data_std == 0] = 1
            lfp = (lfp - lfp.mean(dim=0)[None, :]) / data_std[None, :]

        save_dict = dict(
            lfp=lfp,
            vel=vel,
            t=t,
            trial_start_inds=trial_start_inds,
            lfp_channel_names=lfp_channel_names,
        )

        if save_data:
            save_path = self.get_processed_raw_data_file_path(
                subject=subject, session=session
            )
            torch.save(save_dict, save_path)
        return save_dict

    def process_single_session_segments(self, subject, session):
        processed_raw_data_path = self.get_processed_raw_data_file_path(
            subject=subject, session=session
        )
        data = torch.load(processed_raw_data_path, weights_only=True)

        lfp = data["lfp"]
        vel = data["vel"]
        lfp_channel_names = data["lfp_channel_names"]
        num_steps = lfp.shape[0]

        lfp = lfp[..., : self.config.d_lfp]

        num_steps_in_segment = int(
            self.config.segment_length / (self.config.delta / 1000)
        )
        num_discard_steps = int(num_steps % num_steps_in_segment)

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
                d_lfp=segmented_lfp.shape[-1],
                d_kinem=segmented_vel.shape[-1],
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
