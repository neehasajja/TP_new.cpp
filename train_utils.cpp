#include <torch/torch.h>
#include <iostream>
#include <map>
#include <vector>

using std::map;
using std::vector;

torch::Tensor sample_trajectories_batch(torch::nn::Module& model, torch::Tensor context, torch::Device device, std::map<std::string, std::map<std::string, torch::Tensor>> cfg) {
    int n_samples = cfg["extractor_cfg"]["n_samples"].item<int>();
    int n_time_steps = cfg["model_params"]["future_num_frames"].item<int>();
    int bs = context.size(0).item<int>();
    torch::Tensor samples = torch::zeros({bs, 1, n_samples, 2 * n_time_steps}, torch::dtype(torch::kFloat32).device(device));
    for (int i = 0; i < n_samples; i++) {
        torch::Tensor z = torch::randn({bs, cfg["cvae_cfg"]["latent_dim"].item<int>()}, torch::dtype(torch::kFloat32).device(device));
        {
            torch::NoGradGuard no_grad;
            torch::Tensor trajectories = model.attr("inference")(z, context).toTensor();
            samples.index_put_({torch::indexing::Slice(), 0, i, torch::indexing::Slice()}, trajectories);
        }
    }
    return samples;
}
