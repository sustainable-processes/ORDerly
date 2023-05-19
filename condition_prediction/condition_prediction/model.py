import tensorflow as tf

TEACHER_FORCE = 0
HARD_SELECTION = 1
SOFT_SELECTION = 2


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
    c1_dim=100,
    s1_dim=100,
    s2_dim=100,
    r1_dim=100,
    r2_dim=100,
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
        input_c1 = tf.keras.layers.Input(shape=(c1_dim,), name="input_c1")
        input_s1 = tf.keras.layers.Input(shape=(s1_dim,), name="input_s1")
        input_s2 = tf.keras.layers.Input(shape=(s2_dim,), name="input_s2")
        input_r1 = tf.keras.layers.Input(shape=(r1_dim,), name="input_r1")
        input_r2 = tf.keras.layers.Input(shape=(r2_dim,), name="input_r2")
    elif (mode == HARD_SELECTION) or (mode == SOFT_SELECTION):
        input_c1 = None
        input_s1 = None
        input_s2 = None
        input_r1 = None
        input_r2 = None
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

    c1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="c1_h1",
    )(h2)
    c1_h1 = add_dropout_and_batchnorm(
        c1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=True,
    )
    c1_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="c1_h2",
    )(c1_h1)
    c1_h2 = add_dropout_and_batchnorm(
        c1_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    c1_output = tf.keras.layers.Dense(c1_dim, activation="softmax", name="c1")(c1_h2)
    input_c1 = do_selection(prev_output=c1_output, true_input=input_c1, mode=mode)

    c1_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="c1_dense")(input_c1)
    c1_dense = add_dropout_and_batchnorm(
        c1_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_c1 = tf.keras.layers.Concatenate(axis=-1, name="concat_fp_c1")(
        [h2, c1_dense]
    )

    s1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s1_h1",
    )(concat_fp_c1)
    s1_h1 = add_dropout_and_batchnorm(
        s1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    s1_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s1_h2",
    )(s1_h1)
    s1_h2 = add_dropout_and_batchnorm(
        s1_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    s1_output = tf.keras.layers.Dense(s1_dim, activation="softmax", name="s1")(s1_h2)
    input_s1 = do_selection(prev_output=s1_output, true_input=input_s1, mode=mode)

    s1_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="s1_dense")(input_s1)
    s1_dense = add_dropout_and_batchnorm(
        s1_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    concat_fp_c1_s1 = tf.keras.layers.Concatenate(axis=-1, name="concat_fp_c1_s1")(
        [h2, c1_dense, s1_dense]
    )

    s2_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s2_h1",
    )(concat_fp_c1_s1)
    s2_h1 = add_dropout_and_batchnorm(
        s2_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    s2_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s2_h2",
    )(s2_h1)
    s2_h2 = add_dropout_and_batchnorm(
        s2_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    s2_output = tf.keras.layers.Dense(s2_dim, activation="softmax", name="s2")(s2_h2)
    input_s2 = do_selection(prev_output=s2_output, true_input=input_s2, mode=mode)

    s2_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="s2_dense")(input_s2)
    s2_dense = add_dropout_and_batchnorm(
        s2_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_c1_s1_s2 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_c1_s1_s2"
    )([h2, c1_dense, s1_dense, s2_dense])

    r1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="r1_h1",
    )(concat_fp_c1_s1_s2)
    r1_h1 = add_dropout_and_batchnorm(
        r1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    r1_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="r1_h2",
    )(r1_h1)
    r1_h2 = add_dropout_and_batchnorm(
        r1_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    r1_output = tf.keras.layers.Dense(r1_dim, activation="softmax", name="r1")(r1_h2)
    input_r1 = do_selection(prev_output=r1_output, true_input=input_r1, mode=mode)

    r1_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="r1_dense")(input_r1)
    r1_dense = add_dropout_and_batchnorm(
        r1_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_c1_s1_s2_r1 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_c1_s1_s2_r1"
    )([h2, c1_dense, s1_dense, s2_dense, r1_dense])

    r2_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="r2_h1",
    )(concat_fp_c1_s1_s2_r1)
    r2_h1 = add_dropout_and_batchnorm(
        r2_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    r2_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="r2_h2",
    )(r2_h1)
    r2_h2 = add_dropout_and_batchnorm(
        r2_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    r2_output = tf.keras.layers.Dense(r2_dim, activation="softmax", name="r2")(r2_h2)
    input_r2 = do_selection(prev_output=r2_output, true_input=input_r2, mode=mode)

    r2_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="r2_dense")(input_r2)
    r2_dense = add_dropout_and_batchnorm(
        r2_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    # just for the purpose of shorter print message
    c1 = c1_output
    s1 = s1_output
    s2 = s2_output
    r1 = r1_output
    r2 = r2_output
    output = [c1, s1, s2, r1, r2]
    if mode == TEACHER_FORCE:
        model = tf.keras.models.Model(
            [input_pfp, input_rxnfp, input_c1, input_s1, input_s2, input_r1, input_r2],
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
        "c1_dense",
        "s1_dense",
        "s2_dense",
        "r1_dense",
        "r2_dense",
        "c1_h1",
        "s1_h1",
        "s2_h1",
        "r1_h1",
        "r2_h1",
        "c1_h2",
        "s1_h2",
        "s2_h2",
        "r1_h2",
        "r2_h2",
        "c1",
        "s1",
        "s2",
        "r1",
        "r2",
    ]
    layers += [i.name for i in to_copy_model.layers if "batchnorm" in i.name]

    for l in layers:
        update_model.get_layer(l).set_weights(to_copy_model.get_layer(l).get_weights())