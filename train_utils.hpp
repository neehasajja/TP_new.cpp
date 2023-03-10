#include <iostream>
#include <vector>
#include <map>
#include <tuple>
 

namespace argparse {
    // define Namespace class
    class Namespace {
    public:
        std::vector<int> selected_frames;
        int bar;

        Namespace(std::vector<int> selected_frames_, int bar_) :
            selected_frames(selected_frames_), bar(bar_) {}
    };

    // define function to parse arguments
    Namespace parse_args(int argc, char* argv[]) {
        // create argument parser and add arguments
        auto parser = argparse::ArgumentParser("Visualize argparse arguments.");
        parser.add_argument("selected_frames")
              .help("selected frames")
              .nargs("+")
              .action([](const std::string& value) {
                return std::stoi(value);
              });

        parser.add_argument("-b", "--bar")
              .help("the number of bar on the output.")
              .set_default(50)
              .action([](const std::string& value) {
                return std::stoi(value);
              });

        // parse arguments
        auto args = parser.parse_args(argc, argv);

        // create Namespace object from parsed arguments
        auto ns = argparse::Namespace(args.get<std::vector<int>>("selected_frames"),
                                      args.get<int>("bar"));

        return ns;
    }

    // define function to print argparse arguments
    void print_argparse_arguments(const argparse::Namespace& p) {
        std::cout << "PARAMETER SETTING" << std::endl;
        std::cout << std::string(p.bar, '-') << std::endl;
        std::map<std::string, std::tuple<int, std::vector<int>>> args = {
            {"selected_frames", std::make_tuple(p.selected_frames.size(), p.selected_frames)},
            {"bar", std::make_tuple(1, std::vector<int>{p.bar})}
        };

        for (const auto& arg : args) {
            std::cout << std::setw(25) << std::left << arg.first << ": ";
            auto arg_tuple = arg.second;
            int size = std::get<0>(arg_tuple);
            auto values = std::get<1>(arg_tuple);

            if (size == 1) {
                std::cout << values[0];
            } else {
                std::cout << std::endl;
                for (int i = 0; i < size; i++) {
                    std::cout << std::string(8, ' ') << values[i] << std::endl;
                }
            }
        }
        std::cout << std::string(p.bar, '-') << std::endl;
    }
}

int main(int argc, char* argv[]) {
    auto args = argparse::parse_args(argc, argv);
    argparse::print_argparse_arguments(args);
    return 0;
}
