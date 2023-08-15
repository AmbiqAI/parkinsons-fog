import os
from typing import cast
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy.typing as npt
import wandb
from .model import fog_model
from .defines import TrainParams, FogParams
from .dataset import FogDataset
from .utils import set_random_seed, setup_logger, env_flag

logger = setup_logger(__name__)


def gen_loss_function():
    """Generate loss function w/ mask (valid + mask)."""
    ce = tf.keras.losses.BinaryCrossentropy(reduction='none')
    def loss_function(real, output, name='loss_function'):
        loss = ce(tf.expand_dims(real[:, :, 0:3], axis=-1), tf.expand_dims(output, axis=-1)) # Example shape (32, 864, 3)
        mask = tf.math.multiply(real[:, :, 3], real[:, :, 4]) # Example shape (32, 864)
        mask = tf.cast(mask, dtype=loss.dtype)
        mask = tf.expand_dims(mask, axis=-1) # Example shape (32, 864, 1)
        mask = tf.tile(mask, multiples=[1, 1, 3]) # Example shape (32, 864, 3)
        loss *= mask # Example shape (32, 864, 3)

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss_function

def train(params: TrainParams):
    """Train FoG model"""
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(project=f"pd-fog", entity="ambiq", dir=params.job_dir)
        wandb.config.update(params.dict())

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        ds = FogDataset(
            ds_path=params.ds_path,
            block_size=params.block_size,
            patch_size=params.patch_size,
        )
        train_ds, val_ds = ds.load_train_datasets(
            num_workers=params.num_workers
        )
        # Shuffle and batch datasets for training
        train_ds = cast(tf.data.Dataset,
            train_ds.shuffle(
                buffer_size=params.buffer_size,
                reshuffle_each_iteration=True,
            )
            .batch(
                batch_size=params.batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        val_ds = val_ds.batch(
            batch_size=params.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        num_classes = 3
        in_shape = (ds.frame_size, num_classes*params.patch_size)
        inputs = tf.keras.Input(in_shape, batch_size=params.batch_size, dtype=tf.float32)

        model = fog_model(
            inputs=inputs,
            params=FogParams(
                model_dim=params.model_dim,
                block_size=params.block_size,
                patch_size=params.patch_size,
                num_encoders=params.num_encoders,
                num_lstms=params.num_lstms,
                num_heads=params.num_heads,
                batch_size=params.batch_size,
                training=True,
                dropout=params.dropout,
            ),
            num_classes=num_classes,
        )

        lr_cycles: int = getattr(params, "lr_cycles", 3)
        scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=params.learning_rate,
            first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
            t_mul=1.65 / (0.1 * lr_cycles * (lr_cycles - 1)),
            m_mul=0.4,
        )
        optimizer = tf.keras.optimizers.Adam(scheduler)
        loss = gen_loss_function()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tfr.keras.metrics.MeanAveragePrecisionMetric(name="map")
        ]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.summary(print_fn=logger.info)

        if params.weights_file:
            model.load_weights(params.weights_file)
        params.weights_file = str(params.job_dir / "model.weights")

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=params.weights_file,
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                save_weights_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv")),
            tf.keras.callbacks.TensorBoard(log_dir=str(params.job_dir), write_steps_per_second=True),
        ]

        try:
            model.fit(
                train_ds,
                epochs=params.epochs,
                steps_per_epoch=params.steps_per_epoch,
                callbacks=model_callbacks,
                validation_data=val_ds
            )
        except KeyboardInterrupt:
            logger.warning("Stopping training due to keyboard interrupt")

        # Restore best weights from checkpoint
        model.load_weights(params.weights_file)

        # Save full model
        tf_model_path = str(params.job_dir / "model.tf")
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

    # END WITH


if __name__ == "__main__":
    train(
        params=TrainParams(
            ds_path="./datasets",
            job_dir="./results/fog"
        )
    )
