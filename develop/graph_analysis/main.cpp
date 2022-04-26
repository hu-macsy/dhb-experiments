#include "lib/CLI11.hpp"

#include <dhb/dynamic_hashed_blocks.h>
#include <dhb/graph.h>

#include <gdsb/batcher.h>
#include <gdsb/experiment.h>
#include <gdsb/graph.h>
#include <gdsb/graph_io.h>
#include <gdsb/timer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::experimental::filesystem;

dhb::Edges create_random_edges(size_t count, unsigned int const max_id, std::mt19937& engine) {
    dhb::Edges random_edges;
    std::uniform_int_distribution<unsigned int> edgeDistribution{0, unsigned(max_id)};
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    for (unsigned int i = 0; i < count; ++i) {
        random_edges.push_back({i % max_id, {edgeDistribution(engine), {distrib(engine), 0u}}});
    }

    std::shuffle(std::begin(random_edges), std::end(random_edges), engine);

    return random_edges;
}

dhb::Edges retrieve_edges(fs::path const& graph_path, bool const temporal_graph,
                          bool const weighted_graph_file, std::mt19937& prng) {
    dhb::Edges edges;
    std::ifstream graph_input(graph_path);
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};

    if (temporal_graph) {
        gdsb::TimestampedEdges<dhb::Edges> timestamped_edges;
        gdsb::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int t) {
                timestamped_edges.edges.push_back(dhb::Edge(u, {v, {distrib(prng), 0u}}));
                timestamped_edges.timestamps.push_back(t);
            });

        timestamped_edges = gdsb::sort(timestamped_edges);
        edges = timestamped_edges.edges;
    } else {
        gdsb::read_graph_unweighted<dhb::Vertex>(graph_input, [&](unsigned int u, unsigned int v) {
            edges.push_back(dhb::Edge(u, {v, {distrib(prng), 0u}}));
        });
    }

    return edges;
}

int main(int argc, char** argv) {
    CLI::App app{"graph_analysis"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");
    std::string graph = "";
    app.add_option("-g,--graph", graph, "Full path to graph file.");
    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");
    size_t batch_size = 0;
    bool temporal_graph = false;
    app.add_flag("-y,--temporal-graph", temporal_graph,
                 "Flag in case the graph file contains timestamps."
                 "This will sort all edges based on the time stamp.");

    CLI11_PARSE(app, argc, argv);

    fs::path graph_path(std::move(graph));
    if (!fs::exists(graph_path)) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::mt19937 prng{42};

    unsigned int edge_count_in = 0;

    dhb::Matrix<dhb::EdgeData> matrix = [&]() {
        dhb::Edges edges = retrieve_edges(graph_path, temporal_graph, weighted_graph_file, prng);

        unsigned int const edge_count_in = edges.size();

        dhb::Matrix<dhb::EdgeData> matrix(std::move(edges));

        return matrix;
    }();

    unsigned int const edge_count_out = matrix.edges_count();
    unsigned int const vertex_count = matrix.vertices_count();

    dhb::Vertex max_degree_node = 0u;
    dhb::Vertex min_degree_node = 0u;
    unsigned long long int acc_degree = matrix.degree(max_degree_node);

    for (dhb::Vertex u = 1; u < matrix.vertices_count(); ++u) {
        dhb::Vertex u_deg = matrix.degree(u);
        acc_degree += u_deg;

        if (u_deg > matrix.degree(max_degree_node)) {
            max_degree_node = u;
        } else if (u_deg < matrix.degree(min_degree_node)) {
            min_degree_node = u;
        }
    }

    unsigned int const max_degree = matrix.degree(max_degree_node);
    unsigned int const min_degree = matrix.degree(min_degree_node);
    double mean_degree = acc_degree / static_cast<double>(matrix.vertices_count());
    mean_degree = std::ceil(mean_degree * 10.0) / 10.0;

    gdsb::out("experiment", experiment_name);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("temporal_graph", temporal_graph);
    gdsb::out("format", "DHB");
    gdsb::out("vertex_count", vertex_count);
    gdsb::out("edge_count_in", edge_count_in);
    gdsb::out("edge_count_out", edge_count_out);
    gdsb::out("max_degree", max_degree);
    gdsb::out("max_degree_node", max_degree_node);
    gdsb::out("min_degree", min_degree);
    gdsb::out("min_degree_node", min_degree_node);
    gdsb::out("mean_degree", mean_degree);

    return 0;
}
