# %%

# %%
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import param_sharing.reactions.get
import param_sharing.reactions.filters
import param_sharing.learn.ohe
import param_sharing.learn.util

import param_sharing.coley_code.model


df = param_sharing.reactions.get.get_reaction_df(
    cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    verbose=True,
)

mask = param_sharing.reactions.filters.get_classified_rxn_data_mask(df)

# unpickle
rxn_diff_fp = np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)
product_fp = np.load("data/ORD_USPTO/USPTO_product_fp.pkl.npy", allow_pickle=True)
# Run all cells in the "Read in data" section to get data_df

assert df.shape[0] == mask.shape[0]
assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

if True:
    df = df[mask]
    rxn_diff_fp = rxn_diff_fp[mask]
    product_fp = product_fp[mask]
    print("  - Applied mask")

assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

rng = np.random.default_rng(12345)


indexes = np.arange(df.shape[0])
rng.shuffle(indexes)

train_test_split = 0.8
train_val_split = 0.5

test_idx = indexes[int(df.shape[0] * train_test_split) :]
train_val_idx = indexes[: int(df.shape[0] * train_test_split)]
train_idx = train_val_idx[: int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split) :]

assert rxn_diff_fp.shape == product_fp.shape

train_product_fp = tf.convert_to_tensor(product_fp[train_idx])
train_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[train_idx])
val_product_fp = tf.convert_to_tensor(product_fp[val_idx])
val_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[val_idx])

train_catalyst, val_catalyst, cat_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["catalyst_0"]].fillna("NULL"),
    train_idx,
    val_idx,
    tensor_func=tf.convert_to_tensor,
)
train_solvent_0, val_solvent_0, sol0_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["solvent_0"]].fillna("NULL"),
    train_idx,
    val_idx,
    tensor_func=tf.convert_to_tensor,
)
train_solvent_1, val_solvent_1, sol1_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["solvent_1"]].fillna("NULL"),
    train_idx,
    val_idx,
    tensor_func=tf.convert_to_tensor,
)
(
    train_reagents_0,
    val_reagents_0,
    reag0_enc,
) = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["reagents_0"]].fillna("NULL"),
    train_idx,
    val_idx,
    tensor_func=tf.convert_to_tensor,
)
(
    train_reagents_1,
    val_reagents_1,
    reag1_enc,
) = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["reagents_1"]].fillna("NULL"),
    train_idx,
    val_idx,
    tensor_func=tf.convert_to_tensor,
)
train_temperature = tf.convert_to_tensor(
    df["temperature_0"].iloc[train_idx].fillna(-1.0)
)
val_temperature = tf.convert_to_tensor(df["temperature_0"].iloc[val_idx].fillna(-1.0))

del df

print("Loaded data")

# %%

x_train_data = (
    train_product_fp,
    train_rxn_diff_fp,
    train_catalyst,
    train_solvent_0,
    train_solvent_1,
    train_reagents_0,
    train_reagents_1,
)

x_train_eval_data = (
    train_product_fp,
    train_rxn_diff_fp,
)

y_train_data = (
    train_catalyst,
    train_solvent_0,
    train_solvent_1,
    train_reagents_0,
    train_reagents_1,
    train_temperature,
)

x_val_data = (
    val_product_fp,
    val_rxn_diff_fp,
    val_catalyst,
    val_solvent_0,
    val_solvent_1,
    val_reagents_0,
    val_reagents_1,
)

x_val_eval_data = (
    val_product_fp,
    val_rxn_diff_fp,
)

y_val_data = (
    val_catalyst,
    val_solvent_0,
    val_solvent_1,
    val_reagents_0,
    val_reagents_1,
    val_temperature,
)

# %%

train_mode = param_sharing.coley_code.model.HARD_SELECTION

model = param_sharing.coley_code.model.build_teacher_forcing_model(
    pfp_len=train_product_fp.shape[-1],
    rxnfp_len=train_rxn_diff_fp.shape[-1],
    c1_dim=train_catalyst.shape[-1],  # TODO check not top 100
    s1_dim=train_solvent_0.shape[-1],
    s2_dim=train_solvent_1.shape[-1],
    r1_dim=train_reagents_0.shape[-1],
    r2_dim=train_reagents_1.shape[-1],
    N_h1=1024,
    N_h2=100,
    l2v=0,  # TODO check what coef they used
    mode=train_mode,
    dropout_prob=0.2,
    use_batchnorm=True,
)

# we use a separate model for prediction because we use a recurrent setup for prediction
# the pred model is only different after the catalyst
pred_model = param_sharing.coley_code.model.build_teacher_forcing_model(
    pfp_len=train_product_fp.shape[-1],
    rxnfp_len=train_rxn_diff_fp.shape[-1],
    c1_dim=train_catalyst.shape[-1],
    s1_dim=train_solvent_0.shape[-1],
    s2_dim=train_solvent_1.shape[-1],
    r1_dim=train_reagents_0.shape[-1],
    r2_dim=train_reagents_1.shape[-1],
    N_h1=1024,
    N_h2=100,
    l2v=0,
    mode=param_sharing.coley_code.model.HARD_SELECTION,
    dropout_prob=0.2,
    use_batchnorm=True,
)


def mse_ignore_na(y_true, y_pred):
    raise ValueError(
        "This function messes up with autograd, currently we are setting the values to zero, in the future probably need to separate this data out"
    )
    return tf.reduce_mean(
        tf.keras.backend.switch(
            tf.math.is_nan(y_true), tf.zeros_like(y_true), (y_pred - y_true) ** 2
        )
    )


model.compile(
    loss=[
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.MeanSquaredError(),
    ],
    loss_weights=[1, 1, 1, 1, 1, 1e-4],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics={
        "c1": [
            "acc",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
        "s1": [
            "acc",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
        "s2": [
            "acc",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
        "r1": [
            "acc",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
        "r2": [
            "acc",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    },
)

# %%

param_sharing.coley_code.model.update_teacher_forcing_model_weights(
    update_model=pred_model, to_copy_model=model
)

cat_pred = model(
    x_train_data
    if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
    else x_train_eval_data
)[0]
print(
    "true",
    (
        pd.Series(train_catalyst.numpy().argmax(axis=1)).value_counts()
        / train_catalyst.shape[0]
    ).iloc[:5],
    sep="\n",
)
print(
    "pred",
    (
        pd.Series(cat_pred.numpy().argmax(axis=1)).value_counts() / cat_pred.shape[0]
    ).iloc[:5],
    sep="\n",
)

cat_pred = pred_model(x_train_eval_data)[0]
print(
    "[eval] true",
    (
        pd.Series(train_catalyst.numpy().argmax(axis=1)).value_counts()
        / train_catalyst.shape[0]
    ).iloc[:5],
    sep="\n",
)
print(
    "[eval] pred",
    (
        pd.Series(cat_pred.numpy().argmax(axis=1)).value_counts() / cat_pred.shape[0]
    ).iloc[:5],
    sep="\n",
)

# %%

h = model.fit(
    x=x_train_data
    if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
    else x_train_eval_data,
    y=y_train_data,
    epochs=20,
    verbose=1,
    batch_size=1024,
    validation_data=(
        x_val_data
        if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
        else x_val_eval_data,
        y_val_data,
    ),
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir=param_sharing.learn.util.log_dir(
                prefix="TF_", comment="_MOREDATA_REG_HARDSELECT"
            )
        ),
    ],
)

# %%

param_sharing.coley_code.model.update_teacher_forcing_model_weights(
    update_model=pred_model, to_copy_model=model
)

cat_pred = model(
    x_train_data
    if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
    else x_train_eval_data
)[0]
print(
    "true",
    (
        pd.Series(train_catalyst.numpy().argmax(axis=1)).value_counts()
        / train_catalyst.shape[0]
    ).iloc[:5],
    sep="\n",
)
print(
    "pred",
    (
        pd.Series(cat_pred.numpy().argmax(axis=1)).value_counts() / cat_pred.shape[0]
    ).iloc[:5],
    sep="\n",
)

cat_pred = pred_model(x_train_eval_data)[0]
print(
    "[eval] true",
    (
        pd.Series(train_catalyst.numpy().argmax(axis=1)).value_counts()
        / train_catalyst.shape[0]
    ).iloc[:5],
    sep="\n",
)
print(
    "[eval] pred",
    (
        pd.Series(cat_pred.numpy().argmax(axis=1)).value_counts() / cat_pred.shape[0]
    ).iloc[:5],
    sep="\n",
)

# %%

plt.plot(h.history["loss"], label="loss")
plt.plot(h.history["val_loss"], label="val_loss")
plt.legend()
# %%
