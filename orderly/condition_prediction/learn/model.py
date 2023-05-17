import typing
import torch


TEACHER_FORCE = 0
HARD_SELECTION = 1
SOFT_SELECTION = 2


@torch.no_grad()
def argmax_matrix(m):
    max_idx = torch.argmax(m, 1, keepdim=True)
    one_hot = torch.FloatTensor(m.shape)
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def forward_selection(pred, pred_true, mode=TEACHER_FORCE):
    if mode == TEACHER_FORCE:
        return pred_true
    elif mode == HARD_SELECTION:
        return argmax_matrix(pred)
    elif mode == SOFT_SELECTION:
        return pred
    else:
        raise NotImplementedError(f"unknown for {mode=}")


class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: typing.List[int],
        output_dim: int,
        hidden_acts: typing.List[torch.nn.Module],
        output_act,
        use_batchnorm,
        dropout_prob,
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        if not isinstance(hidden_acts, list):
            hidden_acts = [hidden_acts] * len(hidden_dims)
        for dim, hidden_act in zip(hidden_dims, hidden_acts):
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(hidden_act())
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(dim))
            if dropout_prob > 0:
                layers.append(torch.nn.Dropout(p=dropout_prob))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        if output_act is torch.nn.Softmax:
            layers.append(output_act(dim=1))
        else:
            layers.append(output_act())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()

        return self.layers(x)


class ColeyBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mid_dim,
        highway_dim,
        upstream_hidden_dims=[],
        upstream_hidden_acts=[],
        upstream_output_act=torch.nn.ReLU,
        downstream_hidden_dims=[300, 300],
        downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
        downstream_output_act=torch.nn.Identity,
        stochastic_mid=False,
        use_batchnorm=False,
        dropout_prob=0.0,
    ) -> None:
        super(ColeyBlock, self).__init__()

        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.highway_dim = highway_dim
        self.stochastic_mid = stochastic_mid

        self.upstream = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=upstream_hidden_dims,
            output_dim=mid_dim,
            hidden_acts=upstream_hidden_acts,
            output_act=upstream_output_act,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        self.downstream = SimpleMLP(
            input_dim=mid_dim + highway_dim,
            hidden_dims=downstream_hidden_dims,
            output_dim=output_dim,
            hidden_acts=downstream_hidden_acts,
            output_act=downstream_output_act,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        *,
        input,
        highway_input=None,
        training=True,
    ):
        mid = self.upstream(input, training=training)
        if self.stochastic_mid:
            mid = torch.nn.Dropout(p=0.5)(mid)
        if highway_input is not None:
            mid = torch.cat((highway_input, mid), dim=1)
        output = self.downstream(mid, training=training)
        return output, mid


class ColeyModel(torch.nn.Module):
    def __init__(
        self,
        product_fp_dim,
        rxn_diff_fp_dim,
        cat_dim,
        sol1_dim,
        sol2_dim,
        reag1_dim,
        reag2_dim,
        temp_dim,
        use_batchnorm=False,
        dropout_prob=0.0,
    ) -> None:
        super(ColeyModel, self).__init__()
        self.product_fp_dim = product_fp_dim
        self.rxn_diff_fp_dim = rxn_diff_fp_dim
        self.cat_dim = cat_dim
        self.sol1_dim = sol1_dim
        self.sol2_dim = sol2_dim
        self.reag1_dim = reag1_dim
        self.reag2_dim = reag2_dim
        self.temp_dim = temp_dim
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        highway_dim = 0
        mid_dim = 1000
        self.cat_block = ColeyBlock(
            input_dim=product_fp_dim + rxn_diff_fp_dim,
            output_dim=cat_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[1000],
            upstream_hidden_acts=[torch.nn.ReLU],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=True,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

        mid_dim = 100
        self.sol1_block = ColeyBlock(
            input_dim=cat_dim,
            output_dim=sol1_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[],
            upstream_hidden_acts=[],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=False,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

        mid_dim = 100
        self.sol2_block = ColeyBlock(
            input_dim=sol1_dim,
            output_dim=sol2_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[],
            upstream_hidden_acts=[],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=False,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

        mid_dim = 100
        self.reag1_block = ColeyBlock(
            input_dim=sol2_dim,
            output_dim=reag1_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[],
            upstream_hidden_acts=[],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=False,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

        mid_dim = 100
        self.reag2_block = ColeyBlock(
            input_dim=reag1_dim,
            output_dim=reag2_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[],
            upstream_hidden_acts=[],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=False,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

        mid_dim = 100
        self.temp_block = ColeyBlock(
            input_dim=reag2_dim,
            output_dim=temp_dim,
            mid_dim=mid_dim,
            highway_dim=highway_dim,
            upstream_hidden_dims=[],
            upstream_hidden_acts=[],
            downstream_hidden_dims=[300, 300],
            downstream_hidden_acts=[torch.nn.ReLU, torch.nn.Tanh],
            downstream_output_act=torch.nn.Identity,
            stochastic_mid=False,
            use_batchnorm=use_batchnorm,
            dropout_prob=dropout_prob,
        )
        highway_dim += mid_dim

    def forward(
        self,
        *,
        product_fp,
        rxn_diff_fp,
        cat,
        sol1,
        sol2,
        reag1,
        reag2,
        training=True,
        mode=TEACHER_FORCE,
    ):
        cat_output, cat_mid = self.cat_block(
            input=torch.cat((product_fp, rxn_diff_fp), dim=1),
            highway_input=None,
            training=training,
        )
        sol1_output, sol1_mid = self.sol1_block(
            input=forward_selection(pred=cat_output, pred_true=cat, mode=mode),
            highway_input=cat_mid,
            training=training,
        )
        sol2_output, sol2_mid = self.sol2_block(
            input=forward_selection(pred=sol1_output, pred_true=sol1, mode=mode),
            highway_input=sol1_mid,
            training=training,
        )
        reag1_output, reag1_mid = self.reag1_block(
            input=forward_selection(pred=sol2_output, pred_true=sol2, mode=mode),
            highway_input=sol2_mid,
            training=training,
        )
        reag2_output, reag2_mid = self.reag2_block(
            input=forward_selection(pred=reag1_output, pred_true=reag1, mode=mode),
            highway_input=reag1_mid,
            training=training,
        )
        temp_output, _ = self.temp_block(
            input=forward_selection(pred=reag2_output, pred_true=reag2, mode=mode),
            highway_input=reag2_mid,
            training=training,
        )
        return (
            cat_output,
            sol1_output,
            sol2_output,
            reag1_output,
            reag2_output,
            temp_output,
        )

    def forward_dict(
        self,
        *,
        data,
        training=True,
        mode=TEACHER_FORCE,
        indexes=slice(None),
    ):
        output = self.forward(
            product_fp=data["product_fp"][indexes],
            rxn_diff_fp=data["rxn_diff_fp"][indexes],
            cat=data["catalyst"][indexes],
            sol1=data["solvent_1"][indexes],
            sol2=data["solvent_2"][indexes],
            reag1=data["reagents_1"][indexes],
            reag2=data["reagents_2"][indexes],
            training=training,
            mode=mode,
        )
        pred = {}
        (
            pred["catalyst"],
            pred["solvent_1"],
            pred["solvent_2"],
            pred["reagents_1"],
            pred["reagents_2"],
            pred["temperature"],
        ) = output
        return pred
