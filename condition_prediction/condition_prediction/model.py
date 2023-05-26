import tensorflow as tf

from condition_prediction.constants import HARD_SELECTION, SOFT_SELECTION, TEACHER_FORCE


def hard_selection(pred):
    return tf.stop_gradient(tf.one_hot(tf.math.argmax(pred, axis=1), pred.shape[1]))


def do_selection(prev_output, true_input, mode):
    if mode == HARD_SELECTION:
        return hard_selection(prev_output)
    elif mode == SOFT_SELECTION:
        return prev_output
    elif mode == TEACHER_FORCE:
        return true_input
    else:
        raise NotImplementedError(f"unknown for {mode=}")


def add_dropout_and_batchnorm(
    layer, dropout_prob=0.0, use_batchnorm=False, force_stochastic=False
):
    if dropout_prob:
        layer = tf.keras.layers.Dropout(
            rate=dropout_prob,
            name=f'{layer.name.replace("/", "_").replace(":", "_")}_dropout',
        )(layer, training=force_stochastic)
    if use_batchnorm:
        layer = tf.keras.layers.BatchNormalization(
            name=f'{layer.name.replace("/", "_").replace(":", "_")}_batchnorm',
        )(layer)
    return layer


def build_teacher_forcing_model(
    pfp_len=2048,
    rxnfp_len=2048,
    mol1_dim=100,
    mol2_dim=100,
    mol3_dim=100,
    mol4_dim=100,
    mol5_dim=100,
    N_h1=1024,
    N_h2=100,
    l2v=0,
    mode=TEACHER_FORCE,
    dropout_prob=0.0,
    use_batchnorm=False,
    stochastic_mid=True,
) -> tf.keras.models.Model:
    input_pfp = tf.keras.layers.Input(shape=(pfp_len,), name="input_pfp")
    input_rxnfp = tf.keras.layers.Input(shape=(rxnfp_len,), name="input_rxnfp")

    if mode == TEACHER_FORCE:
        input_mol1 = tf.keras.layers.Input(shape=(mol1_dim,), name="input_mol1")
        input_mol2 = tf.keras.layers.Input(shape=(mol2_dim,), name="input_mol2")
        input_mol3 = tf.keras.layers.Input(shape=(mol3_dim,), name="input_mol3")
        input_mol4 = tf.keras.layers.Input(shape=(mol4_dim,), name="input_mol4")
        input_mol5 = tf.keras.layers.Input(shape=(mol5_dim,), name="input_mol5")
    elif (mode == HARD_SELECTION) or (mode == SOFT_SELECTION):
        input_mol1 = None
        input_mol2 = None
        input_mol3 = None
        input_mol4 = None
        input_mol5 = None
    else:
        raise NotImplementedError(f"unknown for {mode=}")

    concat_fp = tf.keras.layers.Concatenate(axis=1)([input_pfp, input_rxnfp])

    h1 = tf.keras.layers.Dense(
        1000,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="fp_transform1",
    )(concat_fp)
    h1 = add_dropout_and_batchnorm(
        h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    h2 = tf.keras.layers.Dense(
        1000,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="fp_transform2",
    )(h1)
    h2 = add_dropout_and_batchnorm(
        h2, dropout_prob=0.5, use_batchnorm=use_batchnorm, force_stochastic=True
    )

    mol1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol1_h1",
    )(h2)
    mol1_h1 = add_dropout_and_batchnorm(
        mol1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=True,
    )
    mol1_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol1_h2",
    )(mol1_h1)
    mol1_h2 = add_dropout_and_batchnorm(
        mol1_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    mol1_output = tf.keras.layers.Dense(mol1_dim, activation="softmax", name="mol1")(
        mol1_h2
    )
    input_mol1 = do_selection(prev_output=mol1_output, true_input=input_mol1, mode=mode)

    mol1_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="mol1_dense")(
        input_mol1
    )
    mol1_dense = add_dropout_and_batchnorm(
        mol1_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_mol1 = tf.keras.layers.Concatenate(axis=-1, name="concat_fp_mol1")(
        [h2, mol1_dense]
    )

    mol2_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol2_h1",
    )(concat_fp_mol1)
    mol2_h1 = add_dropout_and_batchnorm(
        mol2_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    mol2_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol2_h2",
    )(mol2_h1)
    mol2_h2 = add_dropout_and_batchnorm(
        mol2_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    mol2_output = tf.keras.layers.Dense(mol2_dim, activation="softmax", name="mol2")(
        mol2_h2
    )
    input_mol2 = do_selection(prev_output=mol2_output, true_input=input_mol2, mode=mode)

    mol2_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="mol2_dense")(
        input_mol2
    )
    mol2_dense = add_dropout_and_batchnorm(
        mol2_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    concat_fp_mol1_mol2 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_mol1_mol2"
    )([h2, mol1_dense, mol2_dense])

    mol3_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol3_h1",
    )(concat_fp_mol1_mol2)
    mol3_h1 = add_dropout_and_batchnorm(
        mol3_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    mol3_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol3_h2",
    )(mol3_h1)
    mol3_h2 = add_dropout_and_batchnorm(
        mol3_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    mol3_output = tf.keras.layers.Dense(mol3_dim, activation="softmax", name="mol3")(
        mol3_h2
    )
    input_mol3 = do_selection(prev_output=mol3_output, true_input=input_mol3, mode=mode)

    mol3_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="mol3_dense")(
        input_mol3
    )
    mol3_dense = add_dropout_and_batchnorm(
        mol3_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_mol1_mol2_mol3 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_mol1_mol2_mol3"
    )([h2, mol1_dense, mol2_dense, mol3_dense])

    mol4_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol4_h1",
    )(concat_fp_mol1_mol2_mol3)
    mol4_h1 = add_dropout_and_batchnorm(
        mol4_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    mol4_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol4_h2",
    )(mol4_h1)
    mol4_h2 = add_dropout_and_batchnorm(
        mol4_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    mol4_output = tf.keras.layers.Dense(mol4_dim, activation="softmax", name="mol4")(
        mol4_h2
    )
    input_mol4 = do_selection(prev_output=mol4_output, true_input=input_mol4, mode=mode)

    mol4_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="mol4_dense")(
        input_mol4
    )
    mol4_dense = add_dropout_and_batchnorm(
        mol4_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_mol1_mol2_mol3_mol4 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_mol1_mol2_mol3_mol4"
    )([h2, mol1_dense, mol2_dense, mol3_dense, mol4_dense])

    mol5_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol5_h1",
    )(concat_fp_mol1_mol2_mol3_mol4)
    mol5_h1 = add_dropout_and_batchnorm(
        mol5_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    mol5_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="mol5_h2",
    )(mol5_h1)
    mol5_h2 = add_dropout_and_batchnorm(
        mol5_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    mol5_output = tf.keras.layers.Dense(mol5_dim, activation="softmax", name="mol5")(
        mol5_h2
    )
    input_mol5 = do_selection(prev_output=mol5_output, true_input=input_mol5, mode=mode)

    mol5_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="mol5_dense")(
        input_mol5
    )
    mol5_dense = add_dropout_and_batchnorm(
        mol5_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    # just for the purpose of shorter print message
    mol1 = mol1_output
    mol2 = mol2_output
    mol3 = mol3_output
    mol4 = mol4_output
    mol5 = mol5_output
    output = [mol1, mol2, mol3, mol4, mol5]
    if mode == TEACHER_FORCE:
        model = tf.keras.models.Model(
            [
                input_pfp,
                input_rxnfp,
                input_mol1,
                input_mol2,
                input_mol3,
                input_mol4,
                input_mol5,
            ],
            output,
        )
    elif mode == HARD_SELECTION or mode == SOFT_SELECTION:
        model = tf.keras.models.Model([input_pfp, input_rxnfp], output)
    else:
        raise NotImplementedError(f"unknown for {mode=}")

    return model


def update_teacher_forcing_model_weights(update_model, to_copy_model):
    layers = [
        "fp_transform1",
        "fp_transform2",
        "mol1_dense",
        "mol2_dense",
        "mol3_dense",
        "mol4_dense",
        "mol1_h1",
        "mol2_h1",
        "mol3_h1",
        "mol4_h1",
        "mol5_h1",
        "mol1_h2",
        "mol2_h2",
        "mol3_h2",
        "mol4_h2",
        "mol5_h2",
        "mol1",
        "mol2",
        "mol3",
        "mol4",
        "mol5",
    ]
    layers += [i.name for i in to_copy_model.layers if "batchnorm" in i.name]

    for l in layers:
        update_model.get_layer(l).set_weights(to_copy_model.get_layer(l).get_weights())
