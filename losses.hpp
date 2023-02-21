#include <torch/torch.h>
#include <cmath>
#include <vector>

float loss_KLD(torch::Tensor mean, torch::Tensor log_var, std::map<std::string, std::map<std::string, float>> cfg, bool batch_mean = true) {
    float betta = cfg["cvae_cfg"]["betta"];
    float KLD = -0.5 * torch::sum(1 + log_var - mean.pow(2) - log_var.exp()).item<float>();
    if (batch_mean) {
        int bs = mean.size(0);
        KLD = KLD / bs;
    }
    return betta * KLD / mean.size(0);
}

torch::Tensor compute_kernel(torch::Tensor x, torch::Tensor y) {
    int x_size = x.size(0);
    int y_size = y.size(0);
    int dim = x.size(1);
    x = x.unsqueeze(1);  // (x_size, 1, dim)
    y = y.unsqueeze(0);  // (1, y_size, dim)
    auto tiled_x = x.expand({x_size, y_size, dim});
    auto tiled_y = y.expand({x_size, y_size, dim});
    auto kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/static_cast<float>(dim);
    return torch::exp(-kernel_input);  // (x_size, y_size)
}

float compute_mmd(torch::Tensor x, torch::Tensor y, std::map<std::string, std::map<std::string, float>> cfg) {
    float betta = cfg["cvae_cfg"]["betta"];
    int z_dim = cfg["cvae_cfg"]["latent_dim"];
    auto prior_samples = torch::randn({200, z_dim});
    auto x_kernel = compute_kernel(x, x);
    auto y_kernel = compute_kernel(y, y);
    auto xy_kernel = compute_kernel(x, y);
    auto mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean();
    return mmd * betta;
}

torch::Tensor neg_multi_log_likelihood_batch(torch::Tensor gt, torch::Tensor pred, torch::Tensor confidences, torch::Tensor avails) {
    // convert to (batch_size, num_modes, future_len, num_coords)
    gt = gt.unsqueeze(1);  // add modes
    avails = avails.unsqueeze(1).unsqueeze(-1);  // add modes and cords
    auto error = torch::sum(((gt - pred) * avails).pow(2), -1);  // reduce coords and use availability
    error = torch::log(confidences) - 0.5 * torch::sum(error, -1);  // reduce time
    auto max_value = std::get<0>(error.max(1, true));  // error are negative at this point, so max() gives the minimum one
    error = -torch::log(torch::sum(torch::exp(error - max_value), -1)) - max_value;
    return error.mean();
}
