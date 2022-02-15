#include <dhb/graph.h>

#include <gdsberiment.h>
#include <gdsb/graph_io.h>
#include <gdsb/timer.h>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/components/DynConnectedComponents.hpp>
#include <networkit/global/DynTriangleCounting.hpp>
#include <networkit/graph/BFS.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

#include "lib/CLI11.hpp"

#include <omp.h>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"nk_algos"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");

    std::string graph_raw_path = "graphs/foodweb-baydry.konect";
    app.add_option("-g,--graph", graph_raw_path, "Full path to graph file.");

    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");

    bool temporal_graph_file = false;
    app.add_flag("-y,--temporal-graph", temporal_graph_file,
                 "Flag in case the graph file contains timestamps."
                 "This will sort all edges based on the time stamp.");

    std::string algo{"bfs"};
    app.add_option("-a,--algo", algo, "Algorithm to execute.");

    CLI11_PARSE(app, argc, argv);

    namespace fs = std::experimental::filesystem;
    fs::path graph_path(std::move(graph_raw_path));
    if (!fs::exists(graph_path)) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        return -1;
    }

    // Read the graph.
    dhb::Edges edges;
    std::mt19937 prng{42};
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    std::ifstream graph_input(graph_path);

    if (temporal_graph_file) {
        std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> temporal;
        gdsb::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int ts) {
                temporal.push_back({u, v, ts});
            });
        std::sort(temporal.begin(), temporal.end(),
                  [&](auto lhs, auto rhs) { return std::get<2>(lhs) < std::get<2>(rhs); });

        for (auto t : temporal) {
            auto w = distrib(prng);
            // Generate a random edge weight in [a, b).
            edges.push_back({std::get<0>(t), {std::get<1>(t), w, 0u}});
        };
    } else {
        gdsb::read_graph_unweighted<dhb::Vertex>(graph_input,
                                                   [&](unsigned int u, unsigned int v) {
                                                       auto w = distrib(prng);
                                                       // Generate a random edge weight in [a, b).
                                                       edges.push_back({u, {v, w, 0u}});
                                                   });
    }

    unsigned int long edges_read_in = edges.size();

    // Convert to a NetworKit graph.
    NetworKit::Graph graph(dhb::graph::vertex_count(edges), true, false);
    if (algo == "bfs" || algo == "csr_bfs" || algo == "spmv" || algo == "csr_spmv") {
        for (dhb::Edge const& edge : edges)
            graph.addEdge(edge.source, edge.target.vertex, edge.target.data.weight);
    }

    Aux::Random::setSeed(42, true);

    NetworKit::CSRMatrix matrix;
    // Convert graph to CSR.
    if (algo == "csr_bfs" || algo == "csr_spmv")
        matrix = NetworKit::CSRMatrix::adjacencyMatrix(graph);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (algo == "bfs") {
        uint64_t sum_of_dists = 0;
        for (size_t i = 0; i < 100; ++i) {
            auto u = NetworKit::GraphTools::randomNode(graph);
            NetworKit::Traversal::BFSfrom(graph, u, [&](auto, auto dist) { sum_of_dists += dist; });
        }
        gdsb::out("sum_of_dists", sum_of_dists);
    } else if (algo == "csr_bfs") {
        // Same approach as NetworKit::Traversal::BFSfrom.
        auto matrixBFSfrom = [&](NetworKit::node r, auto f) {
            std::vector<bool> marked(matrix.numberOfRows());
            std::queue<NetworKit::node> q, qNext;
            NetworKit::count dist = 0;
            // enqueue start nodes
            q.push(r);
            marked[r] = true;
            do {
                const auto u = q.front();
                q.pop();
                // apply function
                f(u, dist);
                matrix.forNonZeroElementsInRow(u, [&](NetworKit::node v, double) {
                    if (!marked[v]) {
                        qNext.push(v);
                        marked[v] = true;
                    }
                });
                if (q.empty() && !qNext.empty()) {
                    q.swap(qNext);
                    ++dist;
                }
            } while (!q.empty());
        };

        uint64_t sum_of_dists = 0;
        for (size_t i = 0; i < 100; ++i) {
            auto u = NetworKit::GraphTools::randomNode(graph);
            matrixBFSfrom(u, [&](auto, auto dist) { sum_of_dists += dist; });
        }
        gdsb::out("sum_of_dists", sum_of_dists);
    } else if (algo == "spmv") {
        std::vector<double> vec;
        std::vector<double> temp;
        vec.resize(graph.upperNodeIdBound());
        temp.resize(graph.upperNodeIdBound());

        for (size_t i = 0; i < 10; ++i) {
            auto u = NetworKit::GraphTools::randomNode(graph);
            std::fill(vec.begin(), vec.end(), 0.0);
            vec[u] = 1;

            graph.forNodes([&](NetworKit::node u) {
                double val = 0.0;
                graph.forNeighborsOf(
                    u, [&](NetworKit::node v, NetworKit::edgeweight ew) { val += ew * vec[v]; });
                temp[u] = val;
            });

            std::swap(vec, temp);
        }

        uint64_t sum_of_vals = 0;
        for (double val : vec)
            sum_of_vals += val;
        gdsb::out("sum_of_vals", sum_of_vals);
    } else if (algo == "csr_spmv") {
        std::vector<double> vec;
        std::vector<double> temp;
        vec.resize(graph.upperNodeIdBound());
        temp.resize(graph.upperNodeIdBound());

        for (size_t i = 0; i < 10; ++i) {
            auto u = NetworKit::GraphTools::randomNode(graph);
            std::fill(vec.begin(), vec.end(), 0.0);
            vec[u] = 1;

            for (size_t u = 0; u < matrix.numberOfRows(); ++u) {
                double val = 0.0;
                matrix.forNonZeroElementsInRow(
                    u, [&](NetworKit::node v, double ew) { val += ew * vec[v]; });
                temp[u] = val;
            }

            std::swap(vec, temp);
        }

        uint64_t sum_of_vals = 0;
        for (double val : vec)
            sum_of_vals += val;
        gdsb::out("sum_of_vals", sum_of_vals);
    } else if (algo == "dyn-cc") {
        constexpr size_t batch_size = 10'000;

        NetworKit::DynConnectedComponents algo(graph);
        std::vector<NetworKit::GraphEvent> batch;
        batch.reserve(batch_size);

        algo.run();

        int k = 0;
        for (size_t i = 0; i < edges.size(); i += batch_size) {
            if (k >= 10) // For now, insert the first few batches only.
                break;
            for (size_t j = i; j < i + batch_size; ++j) {
                if (j >= edges.size())
                    break;
                auto u = edges[j].source;
                auto v = edges[j].target.vertex;
                if (graph.hasEdge(u, v))
                    continue;
                graph.addEdge(u, v);
                batch.emplace_back(NetworKit::GraphEvent::EDGE_ADDITION, u, v);
            }

            algo.updateBatch(batch);
            batch.clear();
            ++k;
        }

        gdsb::out("num_components", algo.numberOfComponents());
    } else if (algo == "dyn-triangle") {
        constexpr size_t batch_size = 100'000;

        NetworKit::DynTriangleCounting algo(graph, true);
        std::vector<NetworKit::GraphEvent> batch;
        batch.reserve(batch_size);

        algo.run();

        int k = 0;
        for (size_t i = 0; i < edges.size(); i += batch_size) {
            if (k >= 30) // For now, insert the first few batches only.
                break;
            for (size_t j = i; j < i + batch_size; ++j) {
                if (j >= edges.size())
                    break;
                auto u = edges[j].source;
                auto v = edges[j].target.vertex;
                if (graph.hasEdge(u, v))
                    continue;
                graph.addEdge(u, v);
                batch.emplace_back(NetworKit::GraphEvent::EDGE_ADDITION, u, v);
            }
            graph.sortEdges();

            algo.updateBatch(batch);
            batch.clear();
            ++k;
        }
        gdsb::out("num_triangles", algo.getTriangleCount());
    } else {
        std::cerr << "Invalid algorithm " << algo << std::endl;
        return -1;
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);

    gdsb::out("experiment", experiment_name);
    gdsb::out("threads", 1);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("edges_read_in", edges_read_in);
    gdsb::out("algo", algo);
    gdsb::out("format", "NetworKit");
    gdsb::out("vertex_count", graph.numberOfNodes());
    gdsb::out("memory_footprint_kb_after_insertion", gdsb::memory_usage_in_kb());
    gdsb::out("edge_count_after", graph.numberOfEdges());
    gdsb::out("duration_ms", duration.count());

    return 0;
}
