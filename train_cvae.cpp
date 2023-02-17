#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;
using namespace torch;

// Define a struct for the loss function
struct LossKLD {
  torch::Tensor means;
  torch::Tensor log_var;
  float cfg;

  LossKLD(torch::Tensor means, torch::Tensor log_var, float cfg)
      : means(means), log_var(log_var), cfg(cfg) {}

  torch::Tensor operator()(torch::Tensor z) {
    // Compute the KLD loss
    auto kld = 0.5 * torch::sum(torch::exp(log_var) + means.pow(2) - 1 - log_var);
    return kld * cfg;
  }
};

struct LossMMD {
  torch::Tensor z;

  LossMMD(torch::Tensor z) : z(z) {}

  torch::Tensor operator()(torch::Tensor z_prime) {
    // Compute MMD loss
    auto diff = z - z_prime;
    auto norm = torch::norm(diff, 2, /*dim=*/1);
    return torch::mean(norm);
  }
};

torch::Tensor forward_cvae(torch::Tensor data, torch::nn::Module model, torch::Device device, float cfg) {
  auto context = data["image"].to(device);
  auto targets = data["target_positions"];  // [bs, 50, 2]
  auto targets_xy = torch::cat({targets.slice(/*dim=*/1, 0, 50), targets.slice(/*dim=*/1, 50, 100)}, /*dim=*/1).to(device);  // [bs, 100]

  // Forward pass
  auto result = model->forward({targets_xy, context});
  auto recon = result[0];
  auto means = result[1];
  auto log_var = result[2];
  auto z = result[3];

  auto rec_loss = torch::mean(torch::mean((recon - targets_xy).pow(2), /*dim=*/1));
  
  if (cfg == 0.f) {
    // Compute KLD loss
    auto kld = LossKLD(means, log_var, cfg);
    return {recon, rec_loss, kld(z)};
  } else {
    // Compute MMD loss
    auto mmd = LossMMD(z);
    return {recon, rec_loss, mmd(z)};
  }
}

void forward_cvae(const torch::Tensor& data, torch::nn::Module& model, const torch::Device& device, const torch::Tensor& cfg)
{
    string criterion = cfg.at("cvae_cfg.vlb_loss").item<string>();
    torch::Tensor context = data.at("image").to(device);
    torch::Tensor targets = data.at("target_positions").to(device);
    torch::Tensor targets_xy = torch::cat({targets.slice(/*dim=*/1, /*start=*/0, /*end=*/50).select(/*dim=*/2, /*index=*/0),
                                           targets.slice(/*dim=*/1, /*start=*/0, /*end=*/50).select(/*dim=*/2, /*index=*/1)},
                                          /*dim=*/1);

    auto outputs = model.forward({targets_xy, context});
    auto recon = outputs["recon"].to(torch::kCPU);
    auto means = outputs["means"].to(torch::kCPU);
    auto log_var = outputs["log_var"].to(torch::kCPU);
    auto z = outputs["z"].to(torch::kCPU);
    auto rec_loss = torch::mean(torch::mean((recon - targets_xy).pow(2), /*dim=*/1));

    float vlb_loss = 0;
    if (criterion == "KLD") {
        vlb_loss = /* implement loss_KLD */0;
    } else if (criterion == "MMD") {
        vlb_loss = /* implement loss_MMD */0;
    }
}

void train_cvae(torch::nn::Module& model, torch::data::DataLoader<>& data_loader, torch::optim::Optimizer& optimizer, const torch::Device& device, const torch::Tensor& cfg)
{
    string checkpoint_path = cfg.at("models_checkpoint_path").item<string>();
    string criterion = cfg.at("cvae_cfg.vlb_loss").item<string>();
    int max_num_steps = cfg.at("train_cvae_params.max_num_steps").item<int>();
    int checkpoint_every_n_steps = cfg.at("train_cvae_params.checkpoint_every_n_steps").item<int>();
    bool plot_mode = cfg.at("plot_mode").item<bool>();

    auto data_iterator = data_loader.begin();
    tqdm progress_bar(cfg["train_cvae_params"]["max_num_steps"].item<int>());

    vector<float> losses_train, recon_losses, vlb_losses;
    for (int i = 0; i < cfg ["train_cvae_params"]["max_num_steps"] (); i++) {
        torch::Tensor data;
        try{ 
          data = *data_iterator;
          ++data_iterator;
        }
        catch(const std::out_of_range &e){
          data_iterator = data_loader.begin();
          data = *data_iterator;
          ++data_iterator;
        }
        model.train();
        torch::set_grad_enabled(true);
        auto results = forward_cvae(data,model,device,cgf);
        auto recon_loss = results.first;
        auto vlb_loss = results.second;
        auto loss = recon_loss + vlb_loss;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        losses_train.push_back(loss.item<float>());
        recon_losses.push_back(recon_loss.item<float>());
        vlb_losses.push_back(vlb_loss.item<float>());

        progress_bar.set_description(f"loss:{loss.item<float>()},loss(avg):{accumulate(losses_train.begin(),losses_train.end(), 0.0) / losses_train.size()}";)})
        if(plot_mode)
        {cvae_training_monitor(recon_losses,vlb_losses, criterion);
        }
}
