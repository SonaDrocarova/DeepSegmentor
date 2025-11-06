import time
import wandb
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im


def evaluate_model(model):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if net is not None:
            net.eval()


def train_model_mode(model):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if net is not None:
            net.train()


def validate_model(model, test_dataset, opt, epoch):
    evaluate_model(model)
    val_losses_total = {}
    num_val_batches = 0

    for val_data in test_dataset:
        model.set_input(val_data)
        model.forward()
        val_losses = model.get_current_losses()
        for k, v in val_losses.items():
            val_losses_total[k] = val_losses_total.get(k, 0.0) + v
        num_val_batches += 1

    avg_val_losses = {f'val_{k}': v / num_val_batches for k, v in val_losses_total.items()}
    print(f'[Validation] Epoch {epoch}, batches {num_val_batches} - ' +
          ' '.join([f'{k}: {v:.3f}' for k, v in avg_val_losses.items()]))
    val_total_loss = sum(avg_val_losses.values())
    wandb.log({
        **avg_val_losses,
        "val_total_loss": val_total_loss,  # ✅ this is what your sweep expects!
        "epoch": epoch
    })

    # wandb.log({**avg_val_losses, "epoch": epoch})
    train_model_mode(model)
    return avg_val_losses


def sweep_train():
    with wandb.init(project="deepsegmentor-sperm") as run:
        opt = TrainOptions().parse()

        # Apply sweep config values to opt
        for key, value in run.config.items():
            if hasattr(opt, key):
                setattr(opt, key, value)
        # run.config.update(vars(opt))  # Optional: sync all args to wandb config

        wandb_run_name = run.name
        dataset = create_dataset(opt)
        dataset_size = len(dataset)
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)
        model.setup(opt)
        total_iters = 0

        opt.phase = 'test'
        test_dataset = create_dataset(opt)

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters(epoch)

                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    wandb.log({
                        **{key: wandb.Image(tensor2im(img_tensor)) for key, img_tensor in visuals.items()},
                        "epoch": epoch,
                        "iteration": total_iters
                    })

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    print(f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) ' +
                          ' '.join([f'{k}: {v:.3f}' for k, v in losses.items()]))
                    wandb.log({
                        **losses,
                        "epoch": epoch,
                        "iteration": total_iters,
                        "comp_time": t_comp,
                        "data_time": t_data
                    })

                iter_data_time = time.time()

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
            avg_val_losses = validate_model(model, test_dataset, opt, epoch)

            # current_val_loss = sum(avg_val_losses.values())
            # if current_val_loss is not None:
            #     if current_val_loss < best_val_loss:
            #         best_val_loss = current_val_loss
            #         early_stop_counter = 0
            #         model.save_networks(f'{wandb_run_name}_best')
            #     else:
            #         early_stop_counter += 1
            #         if early_stop_counter >= 10:
            #             print("Early stopping triggered.")
            #             break


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",  # "random" or "grid"
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "val_total_loss"},
        "parameters": {
            "batch_size": {"values": [2, 4, 8]},
            "niter": {"values": [30, 50, 70]},
            "lr_decay_iters": {"values": [60, 80,100, 150]},
            "use_augment": {"values": [True, False]},
            "no_flip": {"values": [0, 1]},
	    "lr":{"values":[0.0001,0.00001,0.000001]},}
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="DeepSegmentor_v2")
    wandb.agent(sweep_id, function=sweep_train, count=1)
