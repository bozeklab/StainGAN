import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer_time import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

model = create_model(opt)           

# visualizer = Visualizer(opt)
total_steps = 0

print("#"*30)
print("TRAINING START", opt.epoch_count)
print("#"*30)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        # visualizer.reset()
        total_steps += 1
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print("ERR ", t)
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)


    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

print("#"*30)
print("TRAINING IS DONE")
print("#"*30)