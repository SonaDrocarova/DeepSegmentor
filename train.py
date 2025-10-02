"""Based one CycleGAN project: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
from util.util import tensor2im  


def evaluate_model(model):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if net is not None:
            net.eval()
def train_model(model):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if net is not None:
            net.train()


def validate_model(model, test_dataset,opt):
            ##############################
        # Validation loop starts here
        ##############################
        evaluate_model(model)  # Switch to eval mode
        val_losses_total = {}
        num_val_batches = 0  # Initialize batch count for averaging

        for val_data in test_dataset:
            model.set_input(val_data)
            model.forward()  # Only forward, no backward or optimization

            val_losses = model.get_current_losses()
            for k, v in val_losses.items():
                val_losses_total[k] = val_losses_total.get(k, 0.0) + v

            num_val_batches += 1

        # Average validation losses
        avg_val_losses = {f'val_{k}': v / num_val_batches for k, v in val_losses_total.items()}

        # Print and log
        print(f'[Validation] Epoch {epoch}, batches {num_val_batches}- ' + ' '.join([f'{k}: {v:.3f}' for k, v in avg_val_losses.items()]))
        wandb.log({
            **avg_val_losses,
            "epoch": epoch
        })

        train_model(model)  # Switch back to train mode

        return avg_val_losses  # Return average validation losses for further processing if needed


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options  # copy options for logging purposes
    wandb.init(
    project="deepsegmentor-sperm",
    # name=opt.name,
    config=vars(opt)
    )
    wandb_run_name = wandb.run.name

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # Create validation (test) dataset
    opt.phase = 'test'  # Ensure proper behavior for validation set
    test_dataset = create_dataset(opt)
    best_val_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()

                # Log each image to wandb
                log_dict = {
                    key: wandb.Image(tensor2im(img_tensor))
                    for key, img_tensor in visuals.items()
                }
                log_dict.update({
                    "epoch": epoch,
                    "iteration": total_iters
                })
                wandb.log(log_dict)
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                    # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                # Print losses to console
                print(f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) ' +
                    ' '.join([f'{k}: {v:.3f}' for k, v in losses.items()]))

                # Log to wandb
                wandb.log({
                    **losses,
                    "epoch": epoch,
                    "iteration": total_iters,
                    "comp_time": t_comp,
                    "data_time": t_data
                })

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        avg_val_losses = validate_model(model, test_dataset,opt)
      
        # # Early stopping logic
        # current_val_loss = sum(avg_val_losses.values())
        # if current_val_loss is not None:
        #     if current_val_loss < best_val_loss:
        #         best_val_loss = current_val_loss
        #         early_stop_counter = 0
        #         model.save_networks(f'{wandb_run_name}_best')  # Save best model
        #     else:
        #         early_stop_counter += 1
        #         if early_stop_counter >= 30:
        #             print("Early stopping triggered.")
        #             break


