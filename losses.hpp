// this is the algorithm that calculates all loss functions for the problem such that it campares the target and predicted output values, also measures how well the nn models the training data.
// this is the code for to compute two types of loss functions for a model, namely KL-divergence (KLD) and Maximum Mean Discrepancy (MMD).

#include <torch/torch.h>
#include <numpy/arrayobject.h>
#include <map>

using namespace torch;


T loss_KLD(const Tensor& mean, const Tensor& log_var, const std::map<std::string, std::map<std::string, T>>& cfg, bool batch_mean = true) {
    T betta = cfg.at("cvae_cfg").at("betta");
    T KLD = -0.5 * torch::sum(1 + log_var - mean.pow(2) - log_var.exp());
    if (batch_mean) {
        int64_t bs = mean.sizes()[0];
        KLD = KLD / bs;
    }
    return betta * KLD / bs;
}


(const Tensor& z, const std::map<std::string, std::map<std::string, T>T loss_MMD>& cfg) {
    T betta = cfg.at("cvae_cfg").at("betta");
    int64_t z_dim = cfg.at("cvae_cfg").at("latent_dim");
    Tensor prior_samples = torch::randn({200, z_dim});
    T MMD = compute_mmd<T>(prior_samples, z);
    return MMD * betta;
}

//The function compute_kernel computes a Gaussian kernel between two inputs, x and y, by computing the mean squared distance between each pair of elements from x and y and taking the negative exponential of the mean//
Tensor compute_kernel(const Tensor& x, const Tensor& y) {
    int64_t x_size = x.sizes()[0];
    int64_t y_size = y.sizes()[0];
    int64_t dim = x.sizes()[1];
    Tensor x_unsqueezed = x.unsqueeze(1);  // (x_size, 1, dim)
    Tensor y_unsqueezed = y.unsqueeze(0);  // (1, y_size, dim)
    Tensor tiled_x = x_unsqueezed.expand({x_size, y_size, dim});
    Tensor tiled_y = y_unsqueezed.expand({x_size, y_size, dim});
    Tensor kernel_input = (tiled_x - tiled_y).pow(2).mean(-1) / static_cast<T>(dim);
    return torch::exp(-kernel_input);  // (x_size, y_size)
}

//The function compute_mmd computes the MMD between two inputs, x and y, by computing the mean of the kernels between x and x, y and y, and x and y, and subtracting the mean of the kernel between x and y from the sum//

T compute_mmd(const Tensor& x, const Tensor& y) {
    Tensor x_kernel = compute_kernel<T>(x, x);
    Tensor y_kernel = compute_kernel<T>(y, y);
    Tensor xy_kernel = compute_kernel<T>(x, y);
    T mmd = x_kernel.mean().item<T>() + y_kernel.mean().item<T>() - 2 * xy_kernel.mean().item<T>();
    return mmd;
}
