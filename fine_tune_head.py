import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im

# ---- Utilities ----

def freeze_module(m):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_module(m):
    for p in m.parameters():
        p.requires_grad = True

def set_bn_eval(m):
    # Keep BN in eval when freezing (no running stats updates)
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                      torch.nn.SyncBatchNorm)):
        m.eval()

def matches_any(name, substrings):
    name_lower = name.lower()
    return any(s in name_lower for s in substrings)

def split_backbone_vs_head(netG):
    """
    Heuristic split:
      - 'backbone' are typical encoder/feature extractor blocks
      - 'head' are segmentation/classifier/fuse/output layers
    Adjust patterns if your model uses different names.
    """
    backbone_like = ["backbone", "encoder", "features", "layer1", "layer2", "layer3", "layer4",
                     "stem", "conv1", "bn1"]
    head_like = ["head", "seg", "classifier", "classifer", "cls", "final", "out", "logit", "score",
                 "side", "fuse", "decoder", "up", "outc", "conv_last"]

    backbone_params, head_params = [], []

    for n, p in netG.named_parameters():
        if matches_any(n, head_like) and not matches_any(n, backbone_like):
            head_params.append(p)
        else:
            # default to backbone unless explicitly head-like
            backbone_params.append(p)

    return backbone_params, head_params

def freeze_backbone_keep_head_trainable(model):
    """Freeze netG backbone (incl. keeping BN eval), leave head trainable."""
    if not hasattr(model, 'netG') or model.netG is None:
        raise RuntimeError("Model doesn't expose netG — adjust this function to your model.")

    netG = model.netG

    # Heuristic split (edit if needed)
    backbone_params, head_params = split_backbone_vs_head(netG)

    # Freeze everything first
    freeze_module(netG)
    netG.apply(set_bn_eval)

    # Then unfreeze head params
    head_param_ids = {id(p) for p in head_params}
    for n, p in netG.named_parameters():
        if id(p) in head_param_ids:
            p.requires_grad = True

    # Return lists for optimizer building + counts for logging
    total = sum(p.numel() for p in netG.parameters())
    trainable = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    return head_params, dict(total_params=total, trainable_params=trainable)

def rebuild_optimizers_head_only(model, opt, head_params):
    """
    Replace model.optimizers with head-only optimizer(s).
    Many CycleGAN/pix2pix-style repos attach optimizers to model.optimizers.
    """
    lr = getattr(opt, "lr", 2e-4)
    beta1 = getattr(opt, "beta1", 0.5)
    wd = getattr(opt, "weight_decay", 0.0)

    # If the repo uses multiple optimizers (e.g., G and D), we keep D untouched
    # but ensure G uses only head params. If you want to fully disable D, set opt.lambda_GAN = 0 in CLI.
    new_opts = []
    # netG head-only
    optG = torch.optim.Adam([p for p in head_params if p.requires_grad], lr=lr, betas=(beta1, 0.999), weight_decay=wd)
    new_opts.append(optG)

    # Try to preserve netD optimizers if present
    if hasattr(model, 'netD') and model.netD is not None:
        # If the original model already built a D optimizer, reuse it; else make a default
        if hasattr(model, 'optimizers') and len(model.optimizers) > 1:
            # assume the second optimizer was for D
            # We will keep it if it still references parameters with grad
            new_opts.append(model.optimizers[1])
        else:
            # Fallback: build a simple Adam for D if you really need it
            d_params = [p for p in model.netD.parameters() if p.requires_grad]
            if len(d_params) > 0:
                optD = torch.optim.Adam(d_params, lr=lr, betas=(beta1, 0.999), weight_decay=wd)
                new_opts.append(optD)

    model.optimizers = new_opts

def print_freeze_summary(tag, counts):
    print(f"[{tag}] Trainable params: {counts['trainable_params']:,} / {counts['total_params']:,} "
          f"({100*counts['trainable_params']/max(1,counts['total_params']):.2f}%)")

# ---- Validation (optional lightweight) ----

def quick_validate(model, dataset, epoch):
    # DeepCrackModel doesn't have .train()/.eval(); it uses the `isTrain` flag.
    was_training = getattr(model, "isTrain", True)
    model.isTrain = False  # switch to inference mode for the wrapper

    losses_total = {}
    n = 0
    with torch.no_grad():
        for data in dataset:
            model.set_input(data)
            # many pix2pix-style repos use `test()` for inference
            if hasattr(model, "test"):
                model.test()
            else:
                model.forward()

            # Some repos don't compute losses in test mode; guard it
            if hasattr(model, "get_current_losses"):
                for k, v in model.get_current_losses().items():
                    try:
                        losses_total[k] = losses_total.get(k, 0.0) + float(v)
                    except Exception:
                        pass
            n += 1

    avg = {f"val_{k}": v / max(1, n) for k, v in losses_total.items()} if losses_total else {}
    if avg:
        print(f"[Validation] epoch {epoch} " + " ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
    else:
        print(f"[Validation] epoch {epoch} (no val losses reported)")

    model.isTrain = was_training  # restore original mode
    return avg


# ---- Main training loop (head-only) ----

if __name__ == "__main__":
    opt = TrainOptions().parse()

    # A few sensible defaults for quick head-only fine-tune
    if not hasattr(opt, "niter") or opt.niter is None:
        opt.niter = 3
    if not hasattr(opt, "niter_decay") or opt.niter_decay is None:
        opt.niter_decay = 0
    if not hasattr(opt, "batch_size") or opt.batch_size is None:
        opt.batch_size = 4

    # Datasets
    train_dataset = create_dataset(opt)
    print(f"The number of training images = {len(train_dataset)}")

    # Build a separate val dataset by switching phase (mirrors your train.py approach)
    val_opt = type("V", (), vars(opt).copy())()
    setattr(val_opt, "phase", "test")
    val_dataset = create_dataset(val_opt)

    # Model
    model = create_model(opt)
    model.setup(opt)  # this usually builds the original optimizers/schedulers etc.

    # Freeze backbone & keep head trainable
    head_params, counts = freeze_backbone_keep_head_trainable(model)
    print_freeze_summary("netG", counts)

    # Rebuild optimizers to only update head
    rebuild_optimizers_head_only(model, opt, head_params)

    # Training
    total_iters = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            iter_start = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            # IMPORTANT: model.optimize_parameters should respect model.optimizers we replaced
            model.optimize_parameters(epoch)

            if total_iters % getattr(opt, "print_freq", 100) == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start) / opt.batch_size
                msg = f"(epoch: {epoch}, iters: {epoch_iter}) " + " ".join(f"{k}: {v:.3f}" for k,v in losses.items())
                print(msg)

        print(f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time: {int(time.time() - epoch_start)} sec")

        # Skip LR scheduler if it's not properly defined (fine-tuning doesn't need decay)
        try:
            if hasattr(model, "update_learning_rate"):
                model.update_learning_rate()
        except AttributeError as e:
            print(f"[Warning] Skipping LR scheduler update: {e}")
        except NotImplementedError:
            print("[Warning] Skipping LR scheduler (NotImplemented).")


        # Quick validation
        quick_validate(model, val_dataset, epoch)

    # Optional: save the head-tuned weights
    if hasattr(model, "save_networks"):
        model.save_networks(f"head_finetuned_epoch{epoch}")
