#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tuple>
#include <vector>

#include "matplotlibcpp.h"
#include "torch/torch.h"
#include "pytorch_lightning/pytorch_lightning.h"
#include "l5kit/visualization/draw_trajectory.hpp"
#include "l5kit/geometry/transform_points.hpp"
#include "l5kit/rasterization/rasterizer_builder.hpp"
#include "l5kit/data/local_data_manager.hpp"
#include "l5kit/data/chunked_dataset.hpp"
#include "l5kit/dataset/agent_dataset.hpp"
#include "l5kit/evaluation/compute_metrics_csv.hpp"
#include "l5kit/evaluation/write_pred_csv.hpp"
#include "l5kit/evaluation/metrics/neg_multi_log_likelihood.hpp"
#include "l5kit/evaluation/metrics/time_displace.hpp"

#include "lyft_loss.hpp"
#include "lyft_models.hpp"
#include "lyft_utils.hpp"


const int VAL_INTERVAL_SAMPLES = 250000;
const int ALL_DATA_SIZE = 198474478;

const std::string CFG_PATH = "../input/lyft-mpred-seresnext26-pretrained/agent_motion_config.yaml";

// for using the same sampling as test dataset agents,
// these two FRAME settings are requried.
// minimum number of frames an agents must have in the past to be picked
const int MIN_FRAME_HISTORY = 0;
// minimum number of frames an agents must have in the future to be picked
const int MIN_FRAME_FUTURE = 10;
const std::tuple<int, int> VAL_SELECTED_FRAME = std::make_tuple(99,);

// output path for test mode
const std::string CSV_PATH = "./submission.csv";

// set random seeds
const int SEED = 42;
std::mt19937 gen(SEED);

int main() {
    // set C++ random seed
    gen.seed(SEED);

    return 0;
}
