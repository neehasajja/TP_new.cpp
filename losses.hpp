#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <iostream>

torch::Tensor pytorch_neg_multi_log_likelihood_batch(
    torch::Tensor gt,
    torch::Tensor pred,
    torch::Tensor confidences,
    torch::Tensor avails,
    float epsilon = 1.0e-8,
    bool is_reduce = true
) {
    TORCH_CHECK(pred.dim() == 4, "expected 3D (MxTxC) array for pred, got ", pred.sizes());
    auto [batch_size, num_modes, future_len, num_coords] = pred.sizes();

    TORCH_CHECK(
        gt.sizes() == torch::IntArrayRef{batch_size, future_len, num_coords},
        "expected 2D (Time x Coords) array for gt, got ", gt.sizes()
    );

    TORCH_CHECK(
        confidences.sizes() == torch::IntArrayRef{batch_size, num_modes},
        "expected 1D (Modes) array for gt, got ", confidences.sizes()
    );

    TORCH_CHECK(
        torch::allclose(torch::sum(confidences, /*dim=*/1), torch::ones({batch_size}, confidences.options())),
        "confidences should sum to 1"
    );

    TORCH_CHECK(
        avails.sizes() == torch::IntArrayRef{batch_size, future_len},
        "expected 1D (Time) array for gt, got ", avails.sizes()
    );

    TORCH_CHECK(torch::isfinite(pred).all(), "invalid value found in pred");
    TORCH_CHECK(torch::isfinite(gt).all(), "invalid value found in gt");
    TORCH_CHECK(torch::isfinite(confidences).all(), "invalid value found in confidences");
    TORCH_CHECK(torch::isfinite(avails).all(), "invalid value found in avails");

    // convert to (batch_size, num_modes, future_len, num_coords)
    gt = gt.unsqueeze(1);  // add modes
    avails = avails.unsqueeze(1).unsqueeze(-1);  // add modes and cords

    // error (batch_size, num_modes, future_len)
    auto error = torch::sum(
        (gt - pred).pow(2) * avails, /*dim=*/-1
    );  // reduce coords and use availability

    auto log_confidences = torch::log(confidences + epsilon);
    auto negative_half_error = -0.5 * error;

    auto likelihood = log_confidences + negative_half_error;
    auto max_values = torch::max(likelihood, /*dim=*/1, /*keepdim=*/true);
    auto exp_likelihood = torch::exp(likelihood - max_values.values);

    auto log_likelihood = -torch::log(torch::sum(exp_likelihood, /*dim=*/1, /*keepdim=*/true)) - max_values.values;

    if (is_reduce) {
        return torch::mean(log_likelihood);
    } else {
        return log_likelihood;
    }
}

torch::Tensor pytorch_neg_multi_log_likelihood_single(
    torch::Tensor gt,
    torch::Tensor pred,
    torch::Tensor avails
) {
    // pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    // create confidence (bs)x(mode=1)
    auto [batch_size, future_len, num_coords] = pred.sizes();
    auto confidences = pred.new_ones({batch_size, 1});
    return pytorch_neg_multi_log_likelihood_batch(
        gt, pred.unsqueeze(1), confidences, avails
    );
}
