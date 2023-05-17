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
    s1_dim=100,
    s2_dim=100,
    a1_dim=100,
    a2_dim=100,
    a3_dim=100,
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
        input_s1 = tf.keras.layers.Input(shape=(s1_dim,), name="input_s1")
        input_s2 = tf.keras.layers.Input(shape=(s2_dim,), name="input_s2")
        input_a1 = tf.keras.layers.Input(shape=(a1_dim,), name="input_a1")
        input_a2 = tf.keras.layers.Input(shape=(a2_dim,), name="input_a2")
        input_a3 = tf.keras.layers.Input(shape=(a3_dim,), name="input_a3")
    elif (mode == HARD_SELECTION) or (mode == SOFT_SELECTION):
        input_s1 = None
        input_s2 = None
        input_a1 = None
        input_a2 = None
        input_a3 = None
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
    ####
    s1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s1_h1",
    )(h2)
    s1_h1 = add_dropout_and_batchnorm(
        s1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=True,
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

    concat_fp_s1 = tf.keras.layers.Concatenate(axis=-1, name="concat_fp_s1")(
        [h2, s1_dense]
    )

    s2_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="s2_h1",
    )(concat_fp_s1)
    
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
    concat_fp_s1_s2 = tf.keras.layers.Concatenate(axis=-1, name="concat_fp_s1_s2")(
        [h2, s1_dense, s2_dense]
    )

    a1_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a1_h1",
    )(concat_fp_s1_s2)
    a1_h1 = add_dropout_and_batchnorm(
        a1_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    a1_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a1_h2",
    )(a1_h1)
    a1_h2 = add_dropout_and_batchnorm(
        a1_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    a1_output = tf.keras.layers.Dense(a1_dim, activation="softmax", name="a1")(a1_h2)
    input_a1 = do_selection(prev_output=a1_output, true_input=input_a1, mode=mode)

    a1_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="a1_dense")(input_a1)
    a1_dense = add_dropout_and_batchnorm(
        a1_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_s1_s2_a1 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_s1_s2_a1"
    )([h2, s1_dense, s2_dense, a1_dense])

    a2_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a2_h1",
    )(concat_fp_s1_s2_a1)
    a2_h1 = add_dropout_and_batchnorm(
        a2_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    a2_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a2_h2",
    )(a2_h1)
    a2_h2 = add_dropout_and_batchnorm(
        a2_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    a2_output = tf.keras.layers.Dense(a2_dim, activation="softmax", name="a2")(a2_h2)
    input_a2 = do_selection(prev_output=a2_output, true_input=input_a2, mode=mode)

    a2_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="a2_dense")(input_a2)
    a2_dense = add_dropout_and_batchnorm(
        a2_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    concat_fp_s1_s2_a1_a2 = tf.keras.layers.Concatenate(
        axis=-1, name="concat_fp_s1_s2_a1_a2"
    )([h2, s1_dense, s2_dense, a1_dense, a2_dense])

    a3_h1 = tf.keras.layers.Dense(
        N_h1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a3_h1",
    )(concat_fp_s1_s2_a1_a2)
    a3_h1 = add_dropout_and_batchnorm(
        a3_h1,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )
    a3_h2 = tf.keras.layers.Dense(
        N_h1,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2v),
        name="a3_h2",
    )(a3_h1)
    a3_h2 = add_dropout_and_batchnorm(
        a3_h2,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )

    a3_output = tf.keras.layers.Dense(a3_dim, activation="softmax", name="a3")(a3_h2)
    input_a3 = do_selection(prev_output=a3_output, true_input=input_a3, mode=mode)

    a3_dense = tf.keras.layers.Dense(N_h2, activation="relu", name="a3_dense")(input_a3)
    a3_dense = add_dropout_and_batchnorm(
        a3_dense,
        dropout_prob=dropout_prob,
        use_batchnorm=use_batchnorm,
        force_stochastic=False,
    )


    # just for the purpose of shorter print message
    s1 = s1_output
    s2 = s2_output
    a1 = a1_output
    a2 = a2_output
    a3 = a3_output
    output = [s1, s2, a1, a2, a3]
    if mode == TEACHER_FORCE:
        model = tf.keras.models.Model(
            [input_pfp, input_rxnfp, input_s1, input_s2, input_a1, input_a2, input_a3],
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
        "s1_dense",
        "s2_dense",
        "a1_dense",
        "a2_dense",
        "a3_dense",
        "s1_h1",
        "s2_h1",
        "a1_h1",
        "a2_h1",
        "a3_h1",
        "s1_h2",
        "s2_h2",
        "a1_h2",
        "a2_h2",
        "a3_h2",
        "s1",
        "s2",
        "a1",
        "a2",
        "a3",
    ]
    layers += [i.name for i in to_copy_model.layers if "batchnorm" in i.name]

    for l in layers:
        update_model.get_layer(l).set_weights(to_copy_model.get_layer(l).get_weights())
