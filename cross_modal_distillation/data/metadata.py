from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch


class Metadata:
    """
    A wrapper for a metadata DataFrame

    This class provide extra functionality over pd.DataFrame and
    abstracts the dependency on pandas dataframe (for the most part.)
    """

    def __init__(
        self,
        metadata_df: Optional[pd.DataFrame] = None,
        load_path: Optional[str] = None,
    ) -> None:
        if metadata_df is not None and load_path is not None:
            raise ValueError("Only one of metadata df or load path should be set")

        if metadata_df is not None:
            self._metadata_df: pd.DataFrame = metadata_df
        else:
            print(f"Loading metadata from {load_path}...")
            self._metadata_df: pd.DataFrame = self.load(load_path)

    @classmethod
    def merge_metadatas(
        cls,
        metadatas: List["Metadata"],
        drop_duplicate: bool = False,
        merge_columns: Union[str, List[str], None] = None,
        keep="first",
    ) -> "Metadata":
        """
        Merge metadata's dataframes
        If drop_duplicate = True, only one row from rows having same `merge_columns` will remain
        based on `keep` strategy. Default to using all columns.
        """
        metadata_dfs = [m._metadata_df for m in metadatas]
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        if drop_duplicate:
            metadata_df = metadata_df.drop_duplicates(subset=merge_columns, keep=keep)
        return Metadata(metadata_df)

    @property
    def columns(self):
        return self._metadata_df.columns

    def concat(self, new_metadata_df: pd.DataFrame):
        self._metadata_df = pd.concat(
            [self._metadata_df, new_metadata_df], ignore_index=True
        )

    def shuffle(self) -> None:
        """Shuffle the metadata table rows"""
        self._metadata_df = self._metadata_df.sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

    def clear(self) -> None:
        """Setting the metadata to empty table"""
        self._metadata_df = self._metadata_df.head(0)

    def is_empty(self) -> bool:
        return len(self._metadata_df) == 0

    def get_row_by_index(self, idx: int) -> pd.Series:
        """Get a metadata table row"""
        return self._metadata_df.iloc[idx]

    def apply_fn_on_all_rows(self, col_name: str, fn: callable) -> pd.Series:
        """Apply a function on each row of the dataframe"""
        return self._metadata_df[col_name].apply(fn)

    def get_unique_values_in_col(self, col_name: str) -> np.ndarray:
        """Get unique values of a columnn"""
        return self._metadata_df[col_name].unique()

    def save(self, path: str) -> None:
        """Save metadata table to csv after converting lists and tuples to strings"""

        def convert_complex_data(val, delimiter=","):
            if isinstance(val, (list, tuple)):
                return delimiter.join(map(str, val))
            elif isinstance(val, (dict, torch.Tensor, np.ndarray)):
                raise TypeError(
                    f"Only columns of type list and tuple can be converted and saved, but received {type(val)}."
                )
            else:
                return val

        metadata_save = deepcopy(self._metadata_df)
        if len(metadata_save) > 0:
            for col in metadata_save.columns:
                metadata_save[col] = metadata_save[col].apply(convert_complex_data)
        metadata_save.to_csv(path, index=False)

    def load(self, path: str) -> pd.DataFrame:
        metadata = pd.read_csv(path)

        def convert_from_string(val, delimiter=","):
            # Check if the value is a list or tuple
            if isinstance(val, str) and delimiter in val:
                # Attempt to convert to a list of floats or ints
                val_split = val.split(delimiter)
                if "." in val_split[0] or "e-" in val_split[0] or "e+" in val_split[0]:
                    dtype = float
                else:
                    dtype = int

                try:
                    converted = list(
                        map(dtype, val_split)
                    )  # or int depending on your use case
                except:
                    converted = list(map(str, val_split))
                return converted if len(converted) > 1 else converted[0]
            return val

        # Apply conversion to each column
        for col in metadata.columns:
            metadata[col] = metadata[col].apply(convert_from_string)
        return metadata

    def rename_cols(self, column_name_dict):
        self._metadata_df.rename(columns=column_name_dict)

    def drop_cols(self, columns):
        self._metadata_df.drop(columns=columns)

    def copy_col(self, copy_col_name, new_col_name):
        if copy_col_name not in self._metadata_df:
            raise KeyError(f"{copy_col_name} does not exist in the metadata dataframe.")
        self._metadata_df[new_col_name] = self._metadata_df[copy_col_name]

    def reduce_based_on_col_value(
        self, col_name: str, value: str, regex: bool = False, inverse: bool = False
    ):
        """
        Filter rows based on `value` of the column `col_name`

        regex: whether to use regex expression (contains) or exact value
        inverse: whether to keep the matching values rows or the rows that do not match
        """
        if not regex:
            indices = self._metadata_df[col_name] == value
        else:
            indices = self._metadata_df[col_name].str.contains(value)

        if inverse:
            indices = ~indices

        self._metadata_df = self._metadata_df[indices].reset_index(drop=True)

    def __len__(self):
        return len(self._metadata_df)

    def _get_column_mapping_dict_from_dataframe(
        self, df: pd.DataFrame, key_col: str, value_col: str
    ):
        """
        Get a dictionary containing `key_col` column values as keys and
        `value_col` column values as values
        """
        unique_keys_index = df.drop_duplicates(subset=key_col, keep="first").index

        keys = df.loc[unique_keys_index, key_col]
        values = df.loc[unique_keys_index, value_col]

        output = dict(zip(keys, values))
        return output

    def get_subject_session_d_input(self) -> dict:
        return self._get_column_mapping_dict_from_dataframe(
            self._metadata_df,
            key_col="subject_session",
            value_col="d_input",
        )
