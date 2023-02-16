#include <torch/torch.h>
#include <matplotlibcpp.h>
#include <iostream>

void cvae_training_monitor(std::vector<double> mse_losses, std::vector<double> vlb_losses, std::string criterion) {
    int fig_size_x = 16, fig_size_y = 6;
    auto fig = plt::figure();
    fig.suptitle("Training Monitor", "fontsize", 16);

    auto ax1 = fig.add_subplot(1, 3, 1);
    auto ax2 = fig.add_subplot(1, 3, 2);
    auto ax3 = fig.add_subplot(1, 3, 3);

    torch::Tensor total_loss = torch::from_blob((void*)(mse_losses.data()), {(long int)mse_losses.size()});
    total_loss += torch::from_blob((void*)(vlb_losses.data()), {(long int)vlb_losses.size()});

    ax1.plot(torch::arange(0, mse_losses.size()), total_loss);
    ax2.plot(torch::arange(0, vlb_losses.size()), torch::from_blob((void*)(mse_losses.data()), {(long int)mse_losses.size()}));
    ax3.plot(torch::arange(0, vlb_losses.size()), torch::from_blob((void*)(vlb_losses.data()), {(long int)vlb_losses.size()}));

    ax1.set_ylabel("MSE + " + criterion + " loss");
    ax1.set_xlabel("Iteration");
    ax1.set_yscale("log");
    ax1.grid(true);

    ax2.set_ylabel("MSE");
    ax2.set_xlabel("Iteration");
    ax2.set_yscale("log");
    ax2.grid(true);

    ax3.set_ylabel(criterion + " loss");
    ax3.set_xlabel("Iteration");
    ax3.set_yscale("log");
    ax3.grid(true);

    plt::show();
}

void extractor_training_monitor(std::vector<double> losses_train) {
    int fig_size_x = 14, fig_size_y = 6;
    auto fig = plt::figure();
    fig.suptitle("Extractor Training Monitor", "fontsize", 16);

    auto ax = fig.add_subplot(1, 1, 1);

    ax.plot(torch::arange(0, losses_train.size()), torch::from_blob((void*)(losses_train.data()), {(long int)losses_train.size()}));
    ax.set_yscale("log");
    ax.grid(true);

    plt::show();
}
