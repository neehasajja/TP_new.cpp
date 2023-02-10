#include <unordered_map>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/models/resnet.h>

#include <map>
#include <vector>
#include <string>


class Embeddings : public torch::nn::Module {
public:
    Embeddings(std::unordered_map<std::string, torch::Tensor> cfg) {
        int num_history_channels = (cfg["model_params"]["history_num_frames"].item<int>() + 1) * 2;
        int num_in_channels = 3 + num_history_channels;

        backbone = torchvision::models::resnet50(torch::nn::ResNetOptions(50).pretrained(true));
        backbone_n_out = 2048;
        n_head = cfg["embed_params"]["n_head"].item<int>();
        emb_dim = cfg["embed_params"]["emb_dim"].item<int>();

        // Adjust input channel for the Lyft data
        backbone->conv1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(num_in_channels, backbone->conv1->options.out_channels)
                .kernel_size(backbone->conv1->options.kernel_size)
                .stride(backbone->conv1->options.stride)
                .padding(backbone->conv1->options.padding)
                .bias(false)
        );

        embeddings = torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(backbone_n_out, n_head)),
            torch::nn::ReLU(),
            torch::nn::Linear(n_head, emb_dim)
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        x = backbone->conv1(x);
        x = backbone->bn1(x);
        x = backbone->relu(x);
        x = backbone->maxpool(x);

        x = backbone->layer1(x);
        x = backbone->layer2(x);
        x = backbone->layer3(x);
        x = backbone->layer4(x);

        x = backbone->avgpool(x);
        x = x.flatten(1);

        auto embeddings = this->embeddings(x);

        return embeddings;
    }

private:
    torch::nn::Sequential embeddings;
    int emb_dim;
    int n_head;
    int backbone_n_out;
    torch::nn::Sequential backbone;
};

class Encoder : public torch::nn::Module {
public:
   Encoder(const std::unordered_map<std::string, torch::Tensor>& cfg) : 
        latent_dim(cfg.at("latent_dim").item<int>()),
        frame_embedding_dim(cfg.at("frame_embedding_dim").item<int>()),
        trajectory_length(cfg.at("trajectory_length").item<int>()),
        layers_dims({trajectory_length + frame_embedding_dim}),
        layers(torch::nn::Sequential()),
        linear_means(torch::nn::Linear(layers_dims.back(), latent_dim)),
        linear_log_var(torch::nn::Linear(layers_dims.back(), latent_ 

class  Decoder : public torch::nn::Module {
 public:
  Decoder(const std::unordered_map<std::string, int>& cfg) : cfg(cfg) {
    trajectory_length = cfg.at("future_num_frames") * 2;
    std::vector<int> layers = {
        cfg.at("latent_dim") + cfg.at("emb_dim")
    };
    layers.insert(layers.end(), cfg.at("decoder_layers").begin(),
                  cfg.at("decoder_layers").end());
    layers_ = torch::nn::Sequential();
    for (int i = 0; i < layers.size() - 1; ++i) {
      layers_->push_back(
          torch::nn::Linear(layers[i], layers[i + 1]));
      layers_->push_back(torch::nn::ReLU());
    }
    reconstruction = torch::nn::Linear(layers.back(), trajectory_length);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor emb) {
    x = torch::cat({x, emb}, /*dim=*/1);
    x = layers_(x);
    x = reconstruction(x);
    return x;
  }

 private:
  int trajectory_length;
  torch::nn::Sequential layers_;
  torch::nn::Linear reconstruction;
  const std::unordered_map<std::string, int>& cfg;
};

class CVAE(nn.Module):
    //Conditional variational auto-encoder is to Perform future trajectory auto-encoding conditioned on frame and history embedding which Learnsthe  distribution P(trajectory | embedding)//
  //
    def __init__(self, cfg: Dict):
        super().__init__()

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.embeddings = Embeddings(cfg)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, context):
        c = self.embeddings(context)
        recon = self.decoder(z, c)
        return recon

    def forward(self, future_trj, context):
        c = self.embeddings(context)
        means, log_var = self.encoder(future_trj, c)
        z = self.reparametrize(means, log_var)
        recon = self.decoder(z, c)
        return recon, means, log_var, z

class TrajectoriesExtractor : public nn::Module {
 public:
  TrajectoriesExtractor(const unordered_map<string, int>& cfg)
      : n_samples_(cfg.at("n_samples")),
        n_head_(cfg.at("n_head")),
        p_drop_(cfg.at("p_drop")),
        n_out_(cfg.at("future_num_frames") * 2),
        n_channels_(cfg.at("n_channels")) {
    extractor_ = nn::Sequential(
        nn::Conv2d(1, n_channels_, {n_samples_, 1}, {1, 1}, {0, 0}),
        nn::ReLU(),
        nn::Conv2d(n_channels_, 1, {1, 1}, {1, 1}, {0, 0}),
        nn::ReLU());

    head1_ = nn::Sequential(
        nn::Linear(200, n_head_),
        nn::ReLU(),
        nn::Dropout(p_drop_),
        nn::Linear(n_head_, n_out_));
    head2_ = nn::Sequential(
        nn::Linear(200, n_head_),
        nn::ReLU(),
        nn::Dropout(p_drop_),
        nn::Linear(n_head_, n_out_));
    head3_ = nn::Sequential(
        nn::Linear(200, n_head_),
        nn::ReLU(),
        nn::Dropout(p_drop_),
        nn::Linear(n_head_, n_out_));
  }

  Tensor forward(const Tensor& x) {
    Tensor f = extractor_->forward(x);
    f = f.flatten(1);
    Tensor x_mean = x.mean(2).view({-1, 100});
    f = torch::cat({f, x_mean}, 1);

    Tensor tr_1 = head1_->forward(f);
    Tensor tr_2 = head2_->forward(f);
    Tensor tr_3 = head3_->forward(f);

    Tensor tr = torch::cat({tr_1, tr_2, tr_3}, 1);
    tr = tr.view({-1, 3, 50, 2});
    return tr;
  }

 private:
  int n_samples_;
  int n_head_;
  float p_drop_;
  int n_out_;
  int n_channels_;
  nn::Sequential extractor_;
  nn::Sequential head1_;
  nn::Sequential head2_;
  nn::Sequential head3_;
};

class TrajectoriesPredictor : public torch::nn::Module {
 public:
  TrajectoriesPredictor(std::shared_ptr<torch::nn::Module> cvae_model,
                        std::shared_ptr<torch::nn::Module> extractor_model,
                        std::unordered_map<std::string, int> cfg,
                        torch::Device device)
      : cvae_model_(cvae_model),
        extractor_model_(extractor_model),
        cfg_(cfg),
        device_(device) {}

  torch::Tensor sample_trajectories_batch(torch::Tensor context) {
    int n_samples = cfg_["extractor_cfg.n_samples"];
    int n_time_steps = cfg_["model_params.future_num_frames"];
    int bs = context.size(0);
    torch::Tensor samples = torch::zeros({bs, 1, n_samples, 2 * n_time_steps});

    for (int i = 0; i < n_samples; ++i) {
      torch::Tensor z = torch::randn({bs, cfg_["cvae_cfg.latent_dim"]}).to(device_);
      torch::Tensor trajectories = cvae_model_->forward(std::make_tuple(z, context)).to(device_);
      samples[{{}, 0, i, {}}] = trajectories;
    }
    return samples;
  }

  torch::Tensor forward(std::unordered_map<std::string, torch::Tensor> x) {
    torch::Tensor context = x["image"].to(device_);
    torch::Tensor trajectories = sample_trajectories_batch(context).to(device_);
    torch::Tensor predictions = extractor_model_->forward(trajectories);
    return predictions;
  }

 private:
  std::shared_ptr<torch::nn::Module> cvae_model_;
  std::shared_ptr<torch::nn::Module> extractor_model_;
  std::unordered_map<std::string, int> cfg_;
  torch::Device device_;
};


