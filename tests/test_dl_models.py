import pytest

torch = pytest.importorskip("torch")
from train.dl_models import (MODEL_SPECS, VAE, build_model, count_params,  # noqa: E402
                             score_module, vae_loss)

pytestmark = pytest.mark.torch
D = 20


def _x(n=64):
    torch.manual_seed(0)
    return torch.randn(n, D)


@pytest.mark.parametrize("kind", ["ae", "dae"])
def test_ae_shapes_and_score(kind):
    m = build_model(kind, D).eval()
    x = _x()
    recon = m(x)
    assert recon.shape == x.shape
    s = score_module(kind, m)(x)
    assert s.shape == (64,)
    torch.testing.assert_close(s, ((recon - x) ** 2).mean(dim=1))


def test_dae_noise_only_in_training():
    m = build_model("dae", D, noise_std=0.5)
    x = _x()
    m.eval()
    torch.testing.assert_close(m(x), m(x))
    m.train()
    torch.manual_seed(1)
    a = m(x)
    torch.manual_seed(2)
    b = m(x)
    assert not torch.allclose(a, b)


def test_vae_eval_is_deterministic_and_uses_mu():
    m = build_model("vae", D).eval()
    x = _x()
    r1, mu, logvar = m(x)
    r2, _, _ = m(x)
    torch.testing.assert_close(r1, r2)
    assert mu.shape == logvar.shape == (64, 16)
    torch.testing.assert_close(score_module("vae", m)(x), ((m.reconstruct_mean(x) - x) ** 2).mean(dim=1))


def test_vae_loss_components():
    x = _x()
    loss, rec, kl = vae_loss(x, torch.zeros_like(x), torch.zeros(64, 16), torch.zeros(64, 16), beta=2.0)
    torch.testing.assert_close(rec, (x ** 2).sum(dim=1).mean())
    torch.testing.assert_close(kl, torch.tensor(0.0))
    torch.testing.assert_close(loss, rec)


def test_mlp_probability():
    m = build_model("mlp", D).eval()
    logits = m(_x())
    assert logits.shape == (64,)
    p = score_module("mlp", m)(_x())
    assert ((p > 0) & (p < 1)).all()


def test_depth_changes_param_count():
    shallow = count_params(build_model("ae", D, hidden=(32,)))
    default = count_params(build_model("ae", D, hidden=(64, 32)))
    deep = count_params(build_model("ae", D, hidden=(128, 64, 32)))
    assert shallow < default < deep


def test_model_specs():
    assert {k: v["name"] for k, v in MODEL_SPECS.items()} == {
        "ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}
    assert isinstance(build_model("vae", D), VAE)


@pytest.mark.parametrize("kind", ["ae", "vae"])
def test_latent_and_checkpoint_roundtrip(tmp_path, kind):
    from train.dl_models import encode_latent, load_checkpoint, save_checkpoint
    m = build_model(kind, D, hidden=(64, 32), latent=8).eval()
    x = _x()
    z = encode_latent(m, x)
    assert z.shape == (64, 8)
    save_checkpoint(tmp_path / "m.pt", kind, m, D, (64, 32), 8, 0.1)
    k2, m2 = load_checkpoint(tmp_path / "m.pt")
    assert k2 == kind and not m2.training
    torch.testing.assert_close(encode_latent(m2, x), z)


def test_mlp_has_no_latent():
    from train.dl_models import encode_latent
    with pytest.raises(ValueError):
        encode_latent(build_model("mlp", D), _x())
