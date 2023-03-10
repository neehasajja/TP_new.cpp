#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/torchvision.h>

using namespace torch;
using namespace torch::nn;
using namespace torchvision::models;
using ModelTypes = torch::nn::AnyModule;
using timm::models::efficientnet::EfficientNet;
using timm::models::resnet::ResNet;

namespace {

    std::map<std::string, int64_t> BACKBONE_OUT = {
        {"efficientnet_b0", 1280},
        {"efficientnet_b1", 1280},
        {"efficientnet_b3", 1536},
        {"seresnext26d_32x4d", 2048}
    };

    using ModelTypes = torch::nn::ModuleHolder<EfficientNet, ResNet>;

    nn::ModulePtr extend_input_channel(ModelTypes& model, int64_t channel_scale = 2) {
        auto w = model.get()->backbone().conv_stem->weight;
        int64_t num_hist = (w.size(1) - 3) / 2 - 1;

        Tensor other_part = w.slice(1, 0, num_hist) * (channel_scale - 1);
        other_part = torch::cat({other_part, w.slice(1, 0, num_hist + 1)}, 1);

        Tensor target_part = w.slice(1, num_hist + 1, -4) * (channel_scale - 1);
        target_part = torch::cat({target_part, w.slice(1, num_hist + 1, -3)}, 1);

        Tensor map_part = w.slice(1, -3, w.size(1));

        Tensor conv1_weight = torch::cat({other_part, target_part, map_part}, 1);
        int64_t num_in_channels = (num_hist * channel_scale + 1) * 2 + 3;

        model.get()->backbone().conv_stem = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(num_in_channels, 
                                     model.get()->backbone().conv_stem->options().out_channels(), 
                                     model.get()->backbone().conv_stem->options().kernel_size())
                .stride(model.get()->backbone().conv_stem->options().stride())
                .padding(model.get()->backbone().conv_stem->options().padding())
                .bias(false)
        );
        model.get()->backbone().conv_stem->weight = nn::Parameter(conv1_weight);
        return model;
    }

torch::nn::AnyModule extend_1st_convw_ch(
    ModelTypes& backbone,
    const std::string& backbone_name,
    const int num_in_channels
) {
    int extend_ch = num_in_channels / 3;
    torch::Tensor w;
    if (backbone_name.find("efficientnet") != std::string::npos) {
        w = backbone.get_parameter("conv_stem.weight").detach();
    } else {
        w = backbone.get_parameter("conv1.0.weight").detach();
    }

    torch::Tensor conv1_weight;
    if (num_in_channels - extend_ch * 3 > 0) {
        conv1_weight = torch::cat(
            {w.repeat({extend_ch, 1, 1, 1}), w.narrow(1, 0, num_in_channels - extend_ch * 3)},
            1
        );
    } else {
        conv1_weight = w.repeat({extend_ch, 1, 1, 1});
    }

    if (backbone_name.find("efficientnet") != std::string::npos) {
        auto conv_stem = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(num_in_channels, backbone.get_parameter("conv_stem.weight").sizes()[1], backbone.get_parameter("conv_stem.weight").sizes()[2])
                .stride(backbone.get_parameter("conv_stem.stride").tolist())
                .padding(backbone.get_parameter("conv_stem.padding").tolist())
                .bias(false)
        );
        conv_stem->weight = torch::nn::Parameter(conv1_weight);
        backbone.register_module("conv_stem", conv_stem);
        backbone.register_module("classifier", torch::nn::Identity());
    } else {
        auto conv1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(num_in_channels, backbone.get_parameter("conv1.0.weight").sizes()[0], backbone.get_parameter("conv1.0.weight").sizes()[2])
                .stride(backbone.get_parameter("conv1.0.stride").tolist())
                .padding(backbone.get_parameter("conv1.0.padding").tolist())
                .bias(false)
        );
        conv1->weight = torch::nn::Parameter(conv1_weight);
        backbone.register_module("conv1.0", conv1);
        backbone.register_module("fc", torch::nn::Identity());
    }

    return backbone;
}

class LyftMultiModelImpl : public nn::Module {
public:
    LyftMultiModelImpl(map<string, map<string, map<string, float>>> cfg, int num_modes = 3, string backbone_name = "efficientnet_b1")
        : future_len(cfg["model_params"]["future_num_frames"]),
          num_modes(num_modes) {
        int num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2;
        int num_in_channels = 3 + num_history_channels;
        backbone = torch::jit::load("path/to/backbone.pt"); // load the backbone model from a .pt file

        // Extend 1st conv layer weight for multi-channel input
        backbone = extend_1st_convw_ch(backbone, backbone_name, num_in_channels);

        // Output shape: batch_sizex50x2
        int num_targets = 2 * future_len;
        num_preds = num_targets * num_modes;

        int backbone_out_features = backbone.get()->output()->type()->expect<ListType>()->elements()[0]->expect<TensorType>()->sizes()[1];

        // Linear layer
        logit = register_module("logit", nn::Linear(backbone_out_features, num_preds + num_modes));
    }

    tuple<Tensor, Tensor> forward(Tensor x) {
        Tensor feature = backbone.forward({x}).toTensor();
        Tensor x_out = logit->forward(feature);
        auto [pred, confidences] = torch::split(x_out, num_preds, 1);

        // pred (batch_size)x(modes)x(time)x(2D coords)
        // confidences (batch_size)x(modes)
        int bs = x_out.size(0);
        pred = pred.view({bs, num_modes, future_len, 2});
        confidences = torch::softmax(confidences, 1);

        return make_tuple(pred, confidences);
    }
