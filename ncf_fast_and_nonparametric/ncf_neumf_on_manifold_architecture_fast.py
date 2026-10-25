# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import torch
import torch.nn as nn


class NeuMFOnManifold(nn.Module):
    def __init__(self,
                 user_num: int,
                 latent_dim: int,
                 factor_num: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 model_type: str = 'NeuMF-end',
                 use_item_projection: bool = True,
                 freeze_item_projection: bool = False,
                 item_projection_init_data: "torch.Tensor | None" = None):
        super().__init__()
        assert model_type in ['MLP', 'GMF', 'NeuMF-end']
        if freeze_item_projection and not use_item_projection:
            raise ValueError(
                "freeze_item_projection=True works through the item "
                "projections (identity for MLP + a fixed PCA map for "
                "GMF) and is incompatible with use_item_projection=False "
                "(no projections at all, Z coordinates sliced directly). "
                "Choose one of the two modes."
            )

        self.model_type = model_type
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_item_projection = use_item_projection
        self.freeze_item_projection = freeze_item_projection
        self.latent_dim = latent_dim

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        mlp_user_dim = factor_num * (2 ** (num_layers - 1))
        self.embed_user_MLP = nn.Embedding(user_num, mlp_user_dim)
        self.mlp_user_dim = mlp_user_dim

        if self.use_item_projection:
            self.item_GMF_linear = nn.Linear(latent_dim, factor_num)
            self.item_MLP_linear = nn.Linear(latent_dim, mlp_user_dim)
        else:
            if latent_dim < max(factor_num, mlp_user_dim):
                raise ValueError(
                    f"use_item_projection=False requires latent_len >= "
                    f"max(factor_num, mlp_user_dim) = {max(factor_num, mlp_user_dim)}, "
                    f"got latent_len={latent_dim}."
                )
            self.item_GMF_linear = None
            self.item_MLP_linear = None

        mlp_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        if self.model_type in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weights(item_projection_init_data)

    def _init_weights(self, item_projection_init_data):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)

        if self.freeze_item_projection:
            assert self.latent_dim == self.mlp_user_dim, (
                f"freeze_item_projection assumes latent_dim ({self.latent_dim}) == "
                f"mlp_user_dim ({self.mlp_user_dim}), so that the MLP branch can use "
                f"an information-preserving identity map."
            )
            with torch.no_grad():
                self.item_MLP_linear.weight.copy_(torch.eye(self.mlp_user_dim))
                self.item_MLP_linear.bias.zero_()

            if item_projection_init_data is None:
                raise ValueError(
                    "freeze_item_projection=True requires "
                    "item_projection_init_data (the item_Z matrix) to "
                    "build the fixed, well-conditioned GMF projection."
                )
            with torch.no_grad():
                z = item_projection_init_data.detach().to(torch.float32)
                z_centered = z - z.mean(dim=0, keepdim=True)
                # Top-factor_num principal directions via SVD: a fixed,
                # informative (not random) dimensionality reduction.
                _, _, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                projection = Vt[:self.factor_num]  # (factor_num, latent_dim)
                self.item_GMF_linear.weight.copy_(projection)
                self.item_GMF_linear.bias.zero_()

            self.item_MLP_linear.weight.requires_grad_(False)
            self.item_MLP_linear.bias.requires_grad_(False)
            self.item_GMF_linear.weight.requires_grad_(False)
            self.item_GMF_linear.bias.requires_grad_(False)
        elif self.use_item_projection:
            nn.init.xavier_uniform_(self.item_GMF_linear.weight)
            nn.init.zeros_(self.item_GMF_linear.bias)
            nn.init.xavier_uniform_(self.item_MLP_linear.weight)
            nn.init.zeros_(self.item_MLP_linear.bias)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user, item, item_Z):
        if self.model_type != 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            z_i = item_Z[item]
            if self.freeze_item_projection or self.use_item_projection:
                embed_item_GMF = self.item_GMF_linear(z_i)
            else:
                embed_item_GMF = z_i[..., :self.factor_num]
            output_GMF = embed_user_GMF * embed_item_GMF
        if self.model_type != 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            z_i = item_Z[item]
            if self.freeze_item_projection or self.use_item_projection:
                embed_item_MLP = self.item_MLP_linear(z_i)
            else:
                embed_item_MLP = z_i[..., :self.mlp_user_dim]
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)
        if self.model_type == 'GMF':
            concat = output_GMF
        elif self.model_type == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), dim=-1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)