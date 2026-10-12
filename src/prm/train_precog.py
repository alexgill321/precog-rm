"""PrecogProtoNet — memory-light ProtoNet trainer (lazy token interleaving).

`ProtoNetPRM.fit_from_embeddings` precomputes one fp32 (2T, 3073) token array per
training rollout and holds all of them for the whole run (~1-8 GB per pool). This
subclass replaces only that: each episode interleaves the ~16 rollouts it actually
touches (~6 MB, rebuilt per episode) directly from the (possibly mmap-backed, fp16)
step-embedding arrays. Everything else — architecture, episode-sampling RNG, loss,
LR schedule, checkpoint format, scoring — is inherited from or identical to
ProtoNetPRM, so checkpoints remain loadable by `ProtoNetPRM.load` and results are
comparable with buffered runs (same episode sequence, same math).

Defaults differ from ProtoNetPRM (both off):
  * no_pos=True         — positional embeddings OFF by default
  * use_type_flag=False — the action/response indicator column is REMOVED entirely:
    tokens are (2T, d_emb), the projection is d_emb -> hidden (with the flag on,
    d_emb+1 -> hidden as in ProtoNetPRM)
Both are persisted in checkpoints and `load` rebuilds the correct shapes; an old
ProtoNetPRM checkpoint (no key) is assumed use_type_flag=True, which is what those
models trained with.

Also folds in the ablation hooks used by notebooks/protonet_ablations.ipynb and
scripts/pool_traj_worker.py:
  * input_mode: "both" | "act_only" | "res_only" (set after __init__; persisted)
  * n_layers=0: encoder bypass (torch's TransformerEncoder cannot run 0 layers)

Note: prefix/support-fraction truncation slices the step arrays before
interleaving — identical to slicing the interleaved buffer for input_mode="both"
(the only mode the buffered implementation handled correctly).
"""
import numpy as np

from src.prm.cluster_sequence import ProtoNetPRM, _GradientReversal


class PrecogProtoNet(ProtoNetPRM):
    input_mode = "both"

    def __init__(self, *args, no_pos=True, use_type_flag=False, distance="euclid",
                 **kwargs):
        super().__init__(*args, no_pos=no_pos, **kwargs)
        self.use_type_flag = use_type_flag
        # "euclid": softmax over -||z-c|| (this codebase's convention, eps-guarded);
        # "sqeuclid": softmax over -||z-c||^2 (Snell et al.; linear logits, no singularity)
        self.distance = distance

    # ---- input representation ----
    def _interleave_rollout(self, act_seq, res_seq):
        mode = getattr(self, "input_mode", "both")
        use_flag = bool(getattr(self, "use_type_flag", False))
        d = act_seq.shape[1]
        if mode == "both":
            if use_flag:
                return super()._interleave_rollout(act_seq, res_seq)
            T = act_seq.shape[0]
            tok = np.empty((2 * T, d), dtype=np.float32)
            tok[0::2] = act_seq
            tok[1::2] = res_seq
            return tok, np.repeat(np.arange(T, dtype=np.int64), 2)
        seq = act_seq if mode == "act_only" else res_seq
        width = d + 1 if use_flag else d
        tok = np.zeros((seq.shape[0], width), dtype=np.float32)
        tok[:, :d] = seq
        if use_flag:
            tok[:, d] = 0.0 if mode == "act_only" else 1.0
        return tok, np.arange(seq.shape[0], dtype=np.int64)

    # ---- inference must score with the same distance rule it trained with ----
    def _logits_from_embed(self, embed):
        if getattr(self, "distance", "euclid") != "sqeuclid":
            return super()._logits_from_embed(embed)
        if self._proto_pass is None:
            return np.array([-1e6, 0.0])
        if self._proto_fail is None:
            return np.array([0.0, -1e6])
        d_pass = float(((embed - self._proto_pass) ** 2).sum())
        d_fail = float(((embed - self._proto_fail) ** 2).sum())
        return np.array([-d_fail, -d_pass])

    # ---- checkpoint round-trip (input width depends on use_type_flag, so the base
    # load — which hardcodes d_input = d_emb + 1 — cannot rebuild flag-off models) ----
    def save(self, path):
        import torch
        super().save(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        payload["config"]["use_type_flag"] = bool(getattr(self, "use_type_flag", False))
        payload["config"]["input_mode"] = getattr(self, "input_mode", "both")
        payload["config"]["distance"] = getattr(self, "distance", "euclid")
        payload["config"]["stepmatch_training"] = bool(
            getattr(self, "stepmatch_training", False))
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path):
        import torch
        import torch.nn as nn

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        use_flag = bool(cfg.get("use_type_flag", True))   # old base ckpts trained WITH it
        prm = cls(
            hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], ffn_dim=cfg.get("ffn_dim", 512),
            dropout=cfg.get("dropout", 0.0), max_steps=cfg.get("max_steps", 100),
            no_pos=cfg.get("no_pos", False), lambda_adv=cfg.get("lambda_adv", 0.0),
            lambda_adv_warmup_epochs=cfg.get("lambda_adv_warmup_epochs", 10),
            use_type_flag=use_flag,
        )
        prm.input_mode = cfg.get("input_mode", "both")
        prm.distance = cfg.get("distance", "euclid")
        prm.stepmatch_training = bool(cfg.get("stepmatch_training", False))
        device = cls._select_device()
        d_emb = cfg["d_emb"]
        d_input = d_emb + 1 if use_flag else d_emb

        proj = nn.Linear(d_input, prm.hidden_dim).to(device)
        pos_emb = nn.Embedding(prm.max_steps, prm.hidden_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prm.hidden_dim, nhead=prm.n_heads,
            dim_feedforward=prm.ffn_dim, dropout=prm.dropout, batch_first=True,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=prm.n_layers, enable_nested_tensor=False,
        ).to(device)
        proj.load_state_dict(ckpt["proj"])
        pos_emb.load_state_dict(ckpt["pos_emb"])
        encoder.load_state_dict(ckpt["encoder"])
        proj.eval(); pos_emb.eval(); encoder.eval()

        prm._proj, prm._pos_emb, prm._encoder = proj, pos_emb, encoder
        prm._d_emb, prm._d_input, prm._device = d_emb, d_input, device
        prm.pooling = cfg.get("pooling", "mean")
        prm.temperature = cfg.get("temperature", 1.0)
        prm.platt_a = cfg.get("platt_a", 1.0)
        prm.platt_b = cfg.get("platt_b", 0.0)
        if "attn_query" in ckpt:
            prm._attn_query = nn.Parameter(ckpt["attn_query"].to(device))
            prm.pooling = "attention"
        if "discriminator" in ckpt:
            sd = ckpt["discriminator"]
            n_out = cfg.get("disc_n_domains") or next(
                v.shape[0] for k, v in reversed(list(sd.items()))
                if k.endswith(".weight"))
            discriminator = nn.Sequential(
                nn.Linear(prm.hidden_dim, 64), nn.ReLU(), nn.Linear(64, n_out),
            ).to(device)
            discriminator.load_state_dict(sd)
            discriminator.eval()
            prm._discriminator = discriminator
            prm._disc_n_domains = n_out
        prm.model = True
        print(f"  PrecogProtoNet loaded from {path} "
              f"(device={device}, use_type_flag={use_flag})")
        return prm

    # ---- n_layers=0 bypass (TransformerEncoder.forward indexes layers[0]) ----
    def _encode_batch(self, tokens_list, positions_list):
        if getattr(self, "n_layers", 1) > 0:
            return super()._encode_batch(tokens_list, positions_list)
        import torch
        B = len(tokens_list)
        max_len = max(t.shape[0] for t in tokens_list)
        d_input = tokens_list[0].shape[1]
        tokens_pad = torch.zeros(B, max_len, d_input, device=self._device)
        pos_pad = torch.zeros(B, max_len, dtype=torch.long, device=self._device)
        mask = torch.ones(B, max_len, dtype=torch.bool, device=self._device)
        for j, (tok, pos) in enumerate(zip(tokens_list, positions_list)):
            L = tok.shape[0]
            tokens_pad[j, :L] = torch.from_numpy(tok).to(self._device)
            pos_pad[j, :L] = torch.from_numpy(
                np.minimum(pos, self.max_steps - 1)).to(self._device)
            mask[j, :L] = False
        h = self._proj(tokens_pad)
        if not getattr(self, "no_pos", False):
            h = h + self._pos_emb(pos_pad)
        valid = (~mask).unsqueeze(-1).float()
        return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    # ---- lazy-token episodic training (port of ProtoNetPRM.fit_from_embeddings) ----
    def fit_from_embeddings(
        self,
        act_sequences,
        res_sequences,
        labels,
        task_ids,
        epoch_callback=None,
        model_ids=None,
    ):
        import math
        import torch
        import torch.nn as nn

        device = self._select_device()
        print(f"  PrecogProtoNet training on device: {device} (lazy tokens)")

        if len(set(labels)) < 2:
            self.model = None
            self.default_score = float(labels[0]) if labels else 0.5
            return

        use_adversarial = (
            model_ids is not None
            and self.lambda_adv > 0
            and len(set(model_ids)) >= 2
        )

        # group rollouts by task; a task needs >=1 rollout of each class
        task_to_indices = {}
        for i, (tid, lab) in enumerate(zip(task_ids, labels)):
            task_to_indices.setdefault(tid, {0: [], 1: []})[lab].append(i)
        eligible_tasks = [tid for tid, ci in task_to_indices.items()
                          if len(ci[0]) >= 1 and len(ci[1]) >= 1]
        if not eligible_tasks:
            print("  WARNING: no eligible tasks for episodic training")
            self.model = None
            self.default_score = 0.5
            return
        print(f"  {len(eligible_tasks)} eligible tasks for episodic training "
              f"(of {len(task_to_indices)} total)")

        d_emb = act_sequences[0].shape[1]
        d_input = d_emb + 1 if bool(getattr(self, "use_type_flag", False)) else d_emb

        proj = nn.Linear(d_input, self.hidden_dim).to(device)
        pos_emb = nn.Embedding(self.max_steps, self.hidden_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=self.n_heads,
            dim_feedforward=self.ffn_dim, dropout=self.dropout, batch_first=True,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers, enable_nested_tensor=False,
        ).to(device)
        self._proj, self._pos_emb, self._encoder = proj, pos_emb, encoder
        self._d_emb, self._d_input, self._device = d_emb, d_input, device

        if self.pooling == "attention":
            self._attn_query = nn.Parameter(
                torch.randn(self.hidden_dim, device=device) * 0.02)
        else:
            self._attn_query = None

        all_params = (list(proj.parameters()) + list(pos_emb.parameters())
                      + list(encoder.parameters()))
        if self._attn_query is not None:
            all_params.append(self._attn_query)

        discriminator = None
        disc_batch_size = 16
        if use_adversarial:
            uniq = sorted(set(int(m) for m in model_ids))
            id2idx = {m: k for k, m in enumerate(uniq)}
            n_domains = len(uniq)
            self._disc_n_domains = n_domains
            discriminator = nn.Sequential(
                nn.Linear(self.hidden_dim, 64), nn.ReLU(), nn.Linear(64, n_domains),
            ).to(device)
            self._discriminator = discriminator
            all_params += list(discriminator.parameters())
            all_model_ids = np.array([id2idx[int(m)] for m in model_ids], dtype=np.int64)
            all_indices = np.arange(len(model_ids))
            counts = {uniq[k]: int((all_model_ids == k).sum()) for k in range(n_domains)}
            print(f"  Adversarial training: lambda_adv={self.lambda_adv}, "
                  f"warmup={self.lambda_adv_warmup_epochs} epochs, "
                  f"{n_domains} domains, counts by model_id: {counts}")

        optimizer = torch.optim.AdamW(all_params, lr=self.lr,
                                      weight_decay=self.weight_decay)
        total_steps = self.epochs * self.episodes_per_epoch
        warmup_steps = int(total_steps * self.warmup_frac)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        rng = np.random.RandomState(42)

        support_fraction = getattr(self, "support_fraction", None)

        def build(i, frac=None, prefix_steps=None):
            """Interleave rollout i on demand; optional truncation to a step count
            (prefix_steps) or fraction (frac) BEFORE interleaving — equivalent to
            slicing the interleaved buffer for input_mode='both'."""
            a, r = act_sequences[i], res_sequences[i]
            T = a.shape[0]
            if frac is not None:
                l = max(1, int(T * frac))
                a, r = a[:l], r[:l]
            elif prefix_steps is not None:
                a, r = a[:prefix_steps], r[:prefix_steps]
            return self._interleave_rollout(a, r)

        for epoch in range(self.epochs):
            proj.train(); pos_emb.train(); encoder.train()
            if discriminator is not None:
                discriminator.train()
            epoch_loss = 0.0
            epoch_disc_loss = 0.0

            if use_adversarial and self.lambda_adv_warmup_epochs > 0:
                lam = self.lambda_adv * min(1.0, epoch / self.lambda_adv_warmup_epochs)
            else:
                lam = self.lambda_adv if use_adversarial else 0.0

            for _ep in range(self.episodes_per_epoch):
                tid = eligible_tasks[rng.randint(len(eligible_tasks))]
                cls_idx = task_to_indices[tid]

                support_idx, query_idx = [], []
                support_labels, query_labels = [], []
                for c in [0, 1]:
                    pool = cls_idx[c]
                    n_need = self.k_support + self.q_query
                    replace = len(pool) < n_need
                    chosen = rng.choice(pool, size=n_need, replace=replace)
                    support_idx.extend(chosen[:self.k_support])
                    support_labels.extend([c] * self.k_support)
                    query_idx.extend(chosen[self.k_support:])
                    query_labels.extend([c] * self.q_query)

                # lazy interleave: only this episode's rollouts materialize
                if getattr(self, "stepmatch_training", False):
                    # Per-length prototypes as the CLASSIFICATION rule (not a data
                    # regime): support is drawn disjointly from queries and stays
                    # conceptually complete; each query is truncated to a uniform
                    # length and scored against prototypes formed from the SUPPORT
                    # ENCODED AT THAT LENGTH (clamped). Gradients flow through both
                    # query and support-prefix encodings. Episode uses
                    # `stepmatch_lengths` distinct lengths (cost control).
                    #
                    # Disjoint resample (kills the self-match artifact of
                    # with-replacement sampling at matched lengths):
                    support_idx, query_idx = [], []
                    support_labels, query_labels = [], []
                    for c in [0, 1]:
                        pool = list(cls_idx[c])
                        ks = min(self.k_support, max(1, len(pool) - 1))
                        sup = list(rng.choice(pool, size=ks, replace=False))
                        rest = [i for i in pool if i not in sup] or pool
                        q = list(rng.choice(rest, size=self.q_query,
                                            replace=len(rest) < self.q_query))
                        support_idx += sup; support_labels += [c] * ks
                        query_idx += q;    query_labels += [c] * self.q_query
                    n_lens = int(getattr(self, "stepmatch_lengths", 4))
                    T_qs = [act_sequences[i].shape[0] for i in query_idx]
                    lens = sorted({int(rng.randint(1, max(T_qs) + 1))
                                   for _ in range(n_lens)})
                    # slot = target progress point; query and support each clamp to
                    # their own T within the slot
                    q_uidx = [j % len(lens) for j in range(len(query_idx))]
                    q_len = [min(lens[u], T) for u, T in zip(q_uidx, T_qs)]
                    s_pairs = [build(si, prefix_steps=min(lu, act_sequences[si].shape[0]))
                               for lu in lens for si in support_idx]
                    q_pairs = [build(i, prefix_steps=l)
                               for i, l in zip(query_idx, q_len)]
                else:
                    s_pairs = [build(i, frac=support_fraction) for i in support_idx]
                    if self.prefix_training:
                        q_pairs = []
                        for i in query_idx:
                            T = act_sequences[i].shape[0]
                            if getattr(self, "prefix_fraction", None) is not None:
                                l = max(1, int(T * self.prefix_fraction))
                            else:
                                l = rng.randint(1, T + 1)
                            q_pairs.append(build(i, prefix_steps=l))
                    else:
                        q_pairs = [build(i) for i in query_idx]

                s_embeds = self._encode_batch([p[0] for p in s_pairs],
                                              [p[1] for p in s_pairs])
                q_embeds = self._encode_batch([p[0] for p in q_pairs],
                                              [p[1] for p in q_pairs])

                s_labels_t = torch.tensor(support_labels, device=device)
                if getattr(self, "stepmatch_training", False):
                    # per-slot prototypes: s_embeds is (n_lens * S, H), slot-major
                    U, S = len(lens), len(support_idx)
                    se = s_embeds.view(U, S, -1)
                    proto_pass = se[:, s_labels_t == 1, :].mean(dim=1)   # (U, H)
                    proto_fail = se[:, s_labels_t == 0, :].mean(dim=1)   # (U, H)
                    ui = torch.tensor(q_uidx, device=device)
                    diff = torch.stack([q_embeds - proto_fail[ui],
                                        q_embeds - proto_pass[ui]], dim=1)  # (Q, 2, H)
                else:
                    proto_pass = s_embeds[s_labels_t == 1].mean(dim=0)
                    proto_fail = s_embeds[s_labels_t == 0].mean(dim=0)
                    prototypes = torch.stack([proto_fail, proto_pass])
                    diff = q_embeds.unsqueeze(1) - prototypes.unsqueeze(0)
                if getattr(self, "distance", "euclid") == "sqeuclid":
                    dists = (diff ** 2).sum(dim=-1)          # Snell et al.; smooth at 0
                else:
                    # eps: grad of sqrt blows up at 0 (deterministic encoders can put
                    # a query exactly on a prototype under with-replacement sampling)
                    dists = ((diff ** 2).sum(dim=-1) + 1e-8).sqrt()
                logits = -dists

                q_labels_t = torch.tensor(query_labels, dtype=torch.long, device=device)
                loss_proto = nn.functional.cross_entropy(logits, q_labels_t)

                loss_disc = torch.tensor(0.0, device=device)
                if use_adversarial and lam > 0:
                    disc_idx = rng.choice(all_indices, size=disc_batch_size,
                                          replace=False)
                    d_pairs = [build(i) for i in disc_idx]
                    d_embeds = self._encode_batch([p[0] for p in d_pairs],
                                                  [p[1] for p in d_pairs])
                    d_reversed = _GradientReversal.apply(d_embeds, lam)
                    d_logits = discriminator(d_reversed)
                    d_targets = torch.tensor(all_model_ids[disc_idx],
                                             dtype=torch.long, device=device)
                    loss_disc = nn.functional.cross_entropy(d_logits, d_targets)

                loss = loss_proto + loss_disc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss_proto.item()
                epoch_disc_loss += loss_disc.item()

            avg_loss = epoch_loss / self.episodes_per_epoch
            if use_adversarial:
                avg_disc = epoch_disc_loss / self.episodes_per_epoch
                print(f"    epoch {epoch+1}/{self.epochs}  "
                      f"proto={avg_loss:.4f}  disc={avg_disc:.4f}  lam={lam:.4f}")
            else:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            if epoch_callback is not None:
                proj.eval(); pos_emb.eval(); encoder.eval()
                if discriminator is not None:
                    discriminator.eval()
                self.model = True
                stop = epoch_callback(self, epoch + 1, avg_loss)
                proj.train(); pos_emb.train(); encoder.train()
                if discriminator is not None:
                    discriminator.train()
                if stop:   # truthy return from the callback requests early stop
                    print(f"    early stop after epoch {epoch + 1}")
                    break

        proj.eval(); pos_emb.eval(); encoder.eval()
        if discriminator is not None:
            discriminator.eval()
        self.model = True
