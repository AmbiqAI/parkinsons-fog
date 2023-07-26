import os
import random
import logging
import functools
from typing import Callable, Generator

import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.typing as npt
import h5py
import sklearn
import sklearn.model_selection
from tqdm import tqdm
from .preprocess import resample_signal, normalize_signal, resample_categorical


logger = logging.getLogger(__name__)

SubjectGenerator = Generator[tuple[str, h5py.Group], None, None]
Preprocessor = Callable[[npt.NDArray], npt.NDArray]
SampleGenerator = Generator[tuple[npt.NDArray, npt.NDArray], None, None]

class FogDataset:

    def __init__(self, ds_path: str, block_size: int, patch_size: int) -> None:
        self.block_size = block_size
        self.patch_size = patch_size
        self.ds_path = os.path.join(ds_path, "tlvmc")

    @property
    def features(self) -> list[str]:
        """Features"""
        return ['AccV', 'AccML', 'AccAP']

    @property
    def targets(self) -> list[str]:
        """Targets"""
        return ['StartHesitation', 'Turn', 'Walking' , 'Valid', 'Task']

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 100

    @property
    def frame_size(self) -> int:
        """Frame size"""
        return self.block_size // self.patch_size

    @property
    def subject_ids(self) -> list[str]:
        """Get subject ids"""
        return [
            # tdcsfog
            '02bc69', '07285e', '082f01', '194d1d', '19ea47', '220a17',
            '231c3b', '242a3e', '24a59d', '251738', '2a39f8', '2c98f7',
            '2d57c2', '301ada', '312788', '31d269', '364459', '3b2403',
            '3b2b7a', '48fd62', '4b39ac', '4ba1d3', '4bb5d0', '4ca9b3',
            '4dc2f8', '4f13b4', '51574c', '516a67', '54ee6e', '59f492',
            '5c0b8a', '66341b', '69cc45', '6a3e93', '743f4e', '7688c1',
            '79011a', '7eb666', '7fcee9', '87174c', '8db7dd', '93f49f',
            '9f85da', 'a03db7', 'a80ae4', 'af82b2', 'b19f77', 'bc3908',
            'c7fee4', 'c85fdf', 'c8e721', 'c95ab0', 'd8836b', 'd9312a',
            'e39bc5', 'e8919c', 'e9fc55', 'eeaff0', 'f2c8aa', 'f62eec',
            'f686f0', 'fa8764',
            # defog
            'a3a1f9', 'ab3b2e', 'a066f4', '413532', '4e6c23', '08de77',
            '3e1d75', '5db717', '387ea0', '3a90e2', '7b2e84', '2874c5',
            '72e2c7', 'e86b6e', '575c60', '72716b', '056372', 'a50c72',
            'bb3387', 'c83ff6', '5d9cae', '0e3d49', '040587', '58e067',
            '5d1cf8', 'c12ab3', 'd79fa0', '8c1f5e', 'f28337', '1d7a0d',
            '12f8d1', 'c92925', '2a7175', '7da72f', '00f674', '1fb9cd',
            'd89567', '7f8949', '8d43d9', '107712', 'ae2d35', 'c56629',
            '710c5e', '1a778d', 'e1f62e'
        ]

    @functools.cached_property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids"""
        return list(filter(lambda x: x not in self.test_subject_ids, self.subject_ids))

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids"""
        return [
            # tdcsfog
            '07285e', '220a17', '54ee6e',
            '312788', '24a59d', '4bb5d0',
            '48fd62', '79011a', '7688c1',
            # defog
            'a3a1f9', '387ea0', 'e86b6e',
            'd89567', '8d43d9', '1a778d'
        ]

    def data_generator(
            self,
            subject_generator: SubjectGenerator,
            samples_per_subject: int = 100,
        ) -> SampleGenerator:
        """Generate samples from subject generator.
        Args:
            subject_generator (SubjectGenerator): Subject generator
            samples_per_subject (int, optional): # samples per subject. Defaults to 100.
        Yields:
            SampleGenerator: Sample generator
        """
        for _, records in subject_generator:
            for _ in range(samples_per_subject):
                # Randomly choose a record
                record = records[np.random.choice(list(records.keys()))]
                if record.shape[0] < self.block_size:
                    continue
                block_start = np.random.randint(record.shape[0] - self.block_size)
                block_end = block_start + self.block_size
                data = record[block_start:block_end]
                data = tf.reshape(data, shape=(self.frame_size, self.patch_size, data.shape[1]))

                # Create input ('AccV', 'AccML', 'AccAP')
                x = data[:, :, 0:3]
                x = tf.reshape(x, shape=(self.frame_size, -1))

                # Create target+mask ('StartHesitation', 'Turn', 'Walking') + ('Valid', 'Task')
                y = data[:, :, 3:8]
                y = tf.transpose(y, perm=[0, 2, 1])
                y = tf.reduce_max(y, axis=-1)
                y = tf.cast(y, tf.int32)

                yield x, y
            # END FOR
        # END FOR

    def uniform_subject_generator(
        self,
        subject_ids: list[str] | None,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> SubjectGenerator:
        """Yield data for each subject in the array.

        Args:
            subject_ids (list[str], optional): Array of subject ids. Defaults to all.
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle subject ids. Defaults to True.

        Returns:
            SubjectGenerator: Subject generator
        """
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_idxs = list(range(len(subject_ids)))
        while True:
            if shuffle:
                random.shuffle(subject_idxs)
            for subject_idx in subject_idxs:
                subject_id = subject_ids[subject_idx]
                with h5py.File(os.path.join(self.ds_path, f"{subject_id.decode('ascii')}.h5"), mode="r") as h5:
                    if "data" not in h5:
                        continue
                    yield subject_id, h5["data"]
                # END WITH
            # END FOR
            if not repeat:
                break
        # END WHILE

    def load_train_datasets(
        self,
        train_subjects: float | None = None,
        val_subjects: float | None = None,
        samples_per_subject: int = 100,
        val_size: int | None = None,
        num_workers: int = 1,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load training and validation TF datasets
        Args:
            train_subjects (float | None, optional): # or proportion of train subjects. Defaults to None.
            val_subjects (float | None, optional): # or proportion of val subjects. Defaults to None.
            samples_per_subject (int, optional): # samples per subject. Defaults to 100.
            val_size (int | None, optional): Validation size.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
        """

        if val_subjects is not None and val_subjects >= 1:
            val_subjects = int(val_subjects)

        samples_per_subject = samples_per_subject or 100

        # Get train subjects
        train_subject_ids = self.train_subject_ids

        # Use subset of training subjects
        if train_subjects is not None:
            num_pts = int(train_subjects) if train_subjects > 1 else int(train_subjects * len(train_subject_ids))
            train_subject_ids = train_subject_ids[:num_pts]
        # END IF

        logger.info("Splitting data into train and validation")
        train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
            train_subject_ids, test_size=val_subjects
        )

        if val_size is None:
            val_size = samples_per_subject * (len(val_subject_ids) - 1)

        train_ds = self._parallelize_dataset(
            subject_ids=train_subject_ids,
            samples_per_subject=samples_per_subject,
            num_workers=num_workers,
            repeat=True,
        )

        logger.info(f"Collecting {val_size} validation samples")
        val_ds = self._parallelize_dataset(
            subject_ids=val_subject_ids,
            samples_per_subject=samples_per_subject,
            repeat=False,
            num_workers=num_workers,
        )
        val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        return train_ds, val_ds

    def load_test_dataset(
        self,
        test_subjects: float | None = None,
        samples_per_subject: int = 100,
        repeat: bool = True,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Load testing datasets
        Args:
            test_subjects (float | None, optional): # or proportion of test subjects. Defaults to None.
            samples_per_subject (int, optional): # samples per subject for testing. Defaults to 100.
            repeat (bool, optional): Restart generator when dataset is exhausted. Defaults to True.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            tf.data.Dataset: Test dataset
        """
        test_subject_ids = self.test_subject_ids

        if test_subjects is not None:
            num_pts = int(test_subjects) if test_subjects > 1 else int(test_subjects * len(test_subject_ids))
            test_subject_ids = test_subject_ids[:num_pts]

        test_ds = self._parallelize_dataset(
            subject_ids=test_subject_ids,
            samples_per_subject=samples_per_subject,
            repeat=repeat,
            num_workers=num_workers,
        )
        return test_ds

    @tf.function
    def _parallelize_dataset(
        self,
        subject_ids: npt.NDArray,
        samples_per_subject: int = 100,
        repeat: bool = False,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Generates datasets for given task in parallel using TF `interleave`

        Args:
            subject_ids (npt.NDArray): List of subject IDs.
            samples_per_subject (int, optional): # Samples per subject. Defaults to 100.
            repeat (bool, optional): Should data generator repeat. Defaults to False.
            num_workers (int, optional): Number of parallel workers. Defaults to 1.
        Returns:
            tf.data.Dataset: Parallelize dataset
        """

        def _make_train_dataset(i, split):
            return self._create_dataset_from_generator(
                subject_ids=subject_ids[i * split : (i + 1) * split],
                samples_per_subject=samples_per_subject,
                repeat=repeat
            )

        if num_workers > len(subject_ids):
            num_workers = len(subject_ids)
        split = len(subject_ids) // num_workers
        datasets = [_make_train_dataset(i, split) for i in range(num_workers)]
        if num_workers <= 1:
            return datasets[0]

        return tf.data.Dataset.from_tensor_slices(datasets).interleave(
            lambda x: x,
            cycle_length=num_workers,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def _output_signature(self):
        """Output signature for dataset generator."""
        return (
            tf.TensorSpec((self.frame_size, len(self.features)*self.patch_size), tf.float32),
            tf.TensorSpec((self.frame_size, len(self.targets)), tf.int32),
        )

    def _create_dataset_from_generator(
        self,
        subject_ids: npt.NDArray,
        samples_per_subject: int = 100,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Creates TF dataset generator for task.

        Args:
            subject_ids (npt.NDArray): subject IDs
            samples_per_subject (int, optional): Samples per subject. Defaults to 100.
            repeat (bool, optional): Repeat. Defaults to True.

        Returns:
            tf.data.Dataset: Dataset generator
        """
        ds_gen = functools.partial(self._dataset_sample_generator)
        dataset = tf.data.Dataset.from_generator(
            generator=ds_gen,
            output_signature=self._output_signature(),
            args=(subject_ids, samples_per_subject, repeat),
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        dataset = dataset.with_options(options)
        return dataset

    def _dataset_sample_generator(
        self,
        subject_ids: npt.NDArray,
        samples_per_subject: int = 100,
        repeat: bool = True,
    ) -> SampleGenerator:
        """Internal sample generator for task.

        Args:
            subject_ids (npt.NDArray): subject IDs
            samples_per_subject (int, optional): Samples per subject. Defaults to 100.
            repeat (bool, optional): Repeat. Defaults to True.

        Returns:
            SampleGenerator: Task sample generator
        """
        subject_generator = self.uniform_subject_generator(subject_ids, repeat=repeat)
        data_generator = self.data_generator(
            subject_generator,
            samples_per_subject=samples_per_subject,
        )

        return data_generator


    def prepare_dataset(self):
        """Prepare dataset
        """
        datasets = ['tdcsfog', 'defog']
        for dataset in datasets:
            metadata_df = pd.read_csv(os.path.join(self.ds_path, f"{dataset}_metadata.csv")).set_index('Id')
            subject_ids = metadata_df.Subject.unique()
            # Create HDF5 file per subject
            for subject_id in tqdm(subject_ids, desc='Preparing'):
                with h5py.File(os.path.join(self.ds_path, f"{subject_id}.h5"), "w") as h5:
                    for record_id in metadata_df[metadata_df.Subject == subject_id].index:
                        record_path = os.path.join(self.ds_path, f"train", dataset, f"{record_id}.csv")
                        if not os.path.isfile(record_path):
                            continue
                        series = pd.read_csv(record_path)
                        if dataset in ['tdcsfog']:
                            fs = 128
                            factor = 1 # m/s^2 -> m/s^2
                            series['Valid'] = True
                            series['Task'] = True
                        else:
                            fs = 100
                            factor = 9.80665 # g -> m/s^2
                        # END IF
                        acc_data = resample_signal(
                            series[['AccV', 'AccML', 'AccAP']].values,
                            sample_rate=fs,
                            target_rate=self.sampling_rate,
                            axis=0
                        )
                        acc_data = normalize_signal(factor*acc_data, axis=0)
                        tgt_data = resample_categorical(
                            series[['StartHesitation', 'Turn', 'Walking', 'Valid', 'Task']].values.astype(np.float32),
                            sample_rate=fs,
                            target_rate=self.sampling_rate,
                            axis=0
                        )
                        data = np.hstack((acc_data, tgt_data))
                        h5.create_dataset(f"/data/{record_id}", data=data, compression="gzip")
                    # END FOR
                # END WITH
            # END FOR
        # END FOR
