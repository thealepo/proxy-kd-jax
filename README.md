# Proxy-KD in JAX

A JAX & Flax NNX implementation of **["Knowledge Distillation of Black-Box Large Language Models" (Chen et al., 2024)](https://arxiv.org/abs/2401.07013)**.

<img width="2609" height="1320" alt="Image" src="https://github.com/user-attachments/assets/495d7607-e6d0-418e-8adc-3f6d9c6b86c7" />

---

This implementation serves as an open-source resource for the knowledge distillation techniques introduced in the paper — namely the use of a **Proxy** model, which is aligned to the **Teacher *(Black-Box Proprietary Model)*** so it can serve as a mimic of the teacher's probability distributions. As far as I can tell there's no public implementation of this paper, so here's one in JAX.

The paper introduces a training paradigm that shows up in two phases:

- **Phase 1: Proxy Alignment.** A Proxy model is trained to minimize two things at once: a DPO-style preference loss over a dataset of `(prompt, black-box response (the winning response), proxy's own response (the losing response))`, alongside a negative log-likelihood that raises the proxy's probability on the teacher's outputs. The DPO reference here is the proxy *from the previous iteration*, not a frozen one. Therefore, the proxy keeps preferring the teacher over its own past self.
- **Phase 2: Student Distillation.** Here, we introduce the student model we actually want to distill into. It minimizes a negative log-likelihood against the black-box outputs (hard labels), *plus* a KL against the distribution the aligned proxy produces (soft labels). Each sample's KL is **weighted** by how well the proxy predicts the teacher's output, so the student leans on the proxy only where the proxy is actually reliable.

The trick that makes this work is: the teacher is black-box (we only see its text), so we can't read its distribution. The proxy is white-box, so once it's aligned we *can* read its full distribution, and so we use it for real soft-label KD.

---

## Assumptions & Limitations

This approach rests on the assumption that we can mimic a probability distribution from pure token responses alone. It showed good results in the paper. That said, there is a sheer overhead of training, with three models in play, plus a whole alignment stage for the proxy before the student even starts.

This repo is a small-scale, from-scratch reference: tiny transformers, a mock black-box teacher, byte-level vocab. It's meant to show the *method* end-to-end, not to reproduce the paper's benchmark numbers. The warm-up SFT phase (Dw) from §3.1 is left out.

---

## Example Usage

Run the full two-phase pipeline (Phase 1 → Phase 2) on a mock setup:

```bash
python example.py
```

This builds three differently-sized models (teacher > proxy > student), aligns the proxy to the teacher, then distills into the student. You can also run each phase on its own:

```bash
python proxy_alignment.py   # Phase 1 only (proxy alignment)
python student_kd.py        # Phase 2 only (student distillation)
```

Model sizes are configurable through `TransformerConfig` — `VOCAB_SIZE` and `SEQ_LEN` must match across all three roles, but `HIDDEN_SIZE`, `N_HEADS`, and `N_LAYERS` can differ (that's how you get a smaller student):

```python
proxy_config   = TransformerConfig(HIDDEN_SIZE=32, N_LAYERS=2)
student_config = TransformerConfig(HIDDEN_SIZE=16, N_LAYERS=1)
```

---

## Folder System and what's in each file

| File | What's in it |
|---|---|
| `transformer.py` | The transformer itself. `TransformerConfig`, attention, MLP, and `CausalLanguageModel` (backbone + LM head). |
| `utils.py` | Shared helpers: `get_token_log_probs`, `autoregressive_generation`, `collection`, and the mock `BlackBoxTeacher`. |
| `proxy_alignment.py` | **Phase 1:** DPO preference loss + NLL, the previous-iteration snapshot, and the proxy training loop. |
| `student_kd.py` | **Phase 2:** weighted dense KL + NLL, the Eq-9 sample weights, and the student training loop. |
| `example.py` | The full pipeline tying both phases together with three different model sizes. |
