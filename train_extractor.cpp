#include <torch/torch.h>
#include <tqdm/tqdm.h>
#include <iostream>
#include <vector>

#include "lyft_motion_prediction/train/train_monitor.h"
#include "lyft_motion_prediction/train/train_utils.h"
#include "lyft_motion_prediction/train/losses.h"

void forward_extractor(
    torch::nn::Module& cvae_model,
    torch::nn::Module& extractor_model,
    torch::Tensor data,
    torch::Device device,
    torch::nn::Module& criterion,
    torch::Tensor confs,
    std::map<std::string, int> cfg) {

  torch::Tensor context = data.to(device);
  torch::Tensor trajectories = sample_trajectories_batch(cvae_model, context, device, cfg).to(device);
  torch::Tensor target_availabilities = data["target_availabilities"].to(device);
  torch::Tensor targets = data["target_positions"].to(device);

  // Forward pass
  torch::Tensor preds = extractor_model->forward(trajectories);
  torch::Tensor loss = criterion->forward(targets, preds, confs, target_availabilities);
}

void train_extractor(
    torch::nn::Module& cvae_model,
    torch::nn::Module& extractor_model,
    torch::data::DataLoader<torch::Tensor> data_loader,
    torch::Tensor confs,
    torch::optim::Optimizer optimizer,
    torch::Device device,
    std::map<std::string, int> cfg,
    bool plot_mode) {

  std::string checkpoint_path = cfg["models_checkpoint_path"];

  auto tr_it = data_loader.begin();
  tqdm progress_bar(cfg["train_extractor_params"]["max_num_steps"]);

  std::vector<float> losses_train;
  std::vector<int> iterations;

  for (int i = 0; i < cfg["train_extractor_params"]["max_num_steps"]; ++i) {
    torch::Tensor data = *tr_it;
    ++tr_it;
    if (tr_it == data_loader.end()) {
      tr_it = data_loader.begin();
    }

    extractor_model->train();
    optimizer.zero_grad();

    // Forward
    torch::Tensor loss, preds;
    forward_extractor(cvae_model, extractor_model, data, device, neg_multi_log_likelihood_batch, confs, cfg);

    // Backward
    loss.backward();
    optimizer.step();

    iterations.push_back(i);
    losses_train.push_back(loss.item().toFloat());  // mean per batch
    progress_bar.set_description(f'loss: {loss.item().toFloat()}, loss(avg): {torch::mean(torch::tensor(loss


