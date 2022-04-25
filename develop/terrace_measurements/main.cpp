#include "lib/CLI11.hpp"

#include <dhb/graph.h>

#include <cppexp/batcher.h>
#include <cppexp/experiment.h>
#include <cppexp/graph.h>
#include <cppexp/graph_io.h>
#include <cppexp/timer.h>

// clang-format off
#include <graph.h>
#include <BFS.h>
// clang-format on

#include <omp.h>

#include <cassert>
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"terrace_measurements"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");

    std::string graph_raw_path = "graphs/foodweb-baydry.konect";
    app.add_option("-g,--graph", graph_raw_path, "Full path to graph file.");

    std::string insertion_routine{"none"};
    app.add_option("-r,--insertion-routine", insertion_routine, "Insertion routine to be used.");

    unsigned int desired_thread_count = 0;
    app.add_option("-t,--desired-thread-count", desired_thread_count, "# of threads.");

    size_t batch_size = 0;
    app.add_option("-b, --batch-size", batch_size,
                   "Size of batch for inserting edges of graph. If set to 0 all edges will be "
                   "inserted one batch.");

    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");
    bool temporal_graph_file = false;
    app.add_flag("-y,--temporal-graph", weighted_graph_file, "Graph file contains weight.");

    unsigned int vertex_count = 0;
    app.add_option("-n,--vertex-count", vertex_count, "Defines the count of vertices.");

    CLI11_PARSE(app, argc, argv);

    namespace fs = std::experimental::filesystem;
    fs::path graph_path(std::move(graph_raw_path));

    if (!fs::exists(graph_path)) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        return -1;
    }

    cppexp::out("threads", omp_get_max_threads());
    if (desired_thread_count && omp_get_max_threads() != desired_thread_count)
        throw std::runtime_error("number of workers does not match desired thread count");
    bool is_parallel = omp_get_max_threads() > 1;

    std::mt19937 prng{42};
    dhb::Edges edges;
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    std::ifstream graph_input(graph_path);
    cppexp::TimestampedEdges<dhb::Edges> timestamped_edges;

    size_t edges_read_in = 0;
    size_t as_nv = 0;
    if (!temporal_graph_file) {
        cppexp::read_graph_unweighted<dhb::Vertex>(graph_input,
                                                   [&](unsigned int u, unsigned int v) {
                                                       auto w = distrib(prng);
                                                       // Generate a random edge weight in [a, b).
                                                       edges.push_back({u, {v, w, 0u}});
                                                   });
        edges_read_in = edges.size();
        as_nv = dhb::graph::vertex_count(edges);
    } else {
        cppexp::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int t) {
                auto w = distrib(prng);
                // Generate a random edge weight in [a, b).
                timestamped_edges.edges.push_back(dhb::Edge(u, {v, {w, 0u}}));
                timestamped_edges.timestamps.push_back(t);
            });

        timestamped_edges = cppexp::sort(timestamped_edges);
        edges = timestamped_edges.edges;
    }

    edges_read_in = edges.size();
    as_nv = dhb::graph::vertex_count(edges);

    std::chrono::milliseconds duration;

    unsigned long long memory_footprint_kb_before_construction = cppexp::memory_usage_in_kb();
    auto before_ctor = std::chrono::high_resolution_clock::now();
    graphstore::Graph graph(as_nv);
    auto after_ctor = std::chrono::high_resolution_clock::now();
    unsigned long long memory_footprint_kb_after_construction = cppexp::memory_usage_in_kb();

    size_t vertex_count_before = graph.get_num_vertices();
    size_t edge_count_before = graph.get_num_edges();

    if (insertion_routine == "temporal_deletions") {
        for (dhb::Edge const& e : edges) {
            graph.add_edge(e.source, e.target.vertex, e.target.data.weight);
        }
    }

    auto collect_random_edges = [&] {
        std::vector<dhb::Edge> collected;
        collected.reserve(edges.size() / 2);

        std::uniform_int_distribution<size_t> edge_distrib(0, edges.size() - 1);
        for (size_t i = 0; i < edges.size() / 2; ++i) {
            auto edge = edges[edge_distrib(prng)];
            edge.target.data.weight = distrib(prng);
            collected.emplace_back(edge);
        }
        return collected;
    };

    auto comp = [](auto el, auto er) -> bool {
        if (el.source != er.source)
            return el.source < er.source;
        return el.target.vertex < er.target.vertex;
    };

    auto do_insert = [&]() {
        for (dhb::Edge const& e : edges) {
            graph.add_edge(e.source, e.target.vertex, e.target.data.weight);
        }
        return true;
    };

    auto do_insert_bulk = [&]() {
        std::sort(edges.begin(), edges.end(), comp);
        std::vector<uint32_t> src(edges.size());
        std::vector<uint32_t> dest(edges.size());
        std::vector<uint32_t> weight(edges.size());
        for (size_t i = 0; i < edges.size(); ++i) {
            src[i] = edges[i].source;
            dest[i] = edges[i].target.vertex;
            weight[i] = edges[i].target.data.weight;
        }
        auto perm = graphstore::get_random_permutation(edges.size());
        graph.add_edge_batch(src.data(), dest.data(), weight.data(), edges.size(), perm);
        return true;
    };

    auto do_insert_batch = [&]() {
        if (!batch_size)
            throw std::runtime_error("batch size required");
        cppexp::Batcher<dhb::Edges> batcher(std::begin(edges), std::end(edges), batch_size);

        cppexp::Batch<dhb::Edges> batch = batcher.next_sorted(comp, 1);
        std::vector<uint32_t> src(batch_size);
        std::vector<uint32_t> dest(batch_size);
        std::vector<uint32_t> weight(batch_size);
        while (batch.begin != batcher.end()) {
            size_t i = 0;
            for (auto it = batch.begin; it != batch.end; ++it) {
                src[i] = it->source;
                dest[i] = it->target.vertex;
                weight[i] = it->target.data.weight;
                ++i;
            }
            auto perm = graphstore::get_random_permutation(i);
            graph.add_edge_batch(src.data(), dest.data(), weight.data(), i, perm);
            batch = batcher.next_sorted(comp, 1);
        }
        return true;
    };

    auto do_temporal_deletions = [&](dhb::Edges const& collected) {
        for (dhb::Edge const& e : collected) {
            graph.remove_edge(e.source, e.target.vertex);
        }
        return true;
    };

    if (insertion_routine == "insert") {
        duration = cppexp::benchmark(do_insert);
    } else if (insertion_routine == "insert_bulk") {
        duration = cppexp::benchmark(do_insert_bulk);
    } else if (insertion_routine == "insert_batch") {
        duration = cppexp::benchmark(do_insert_batch);
    } else if (insertion_routine == "temporal_insertions") {
        // Temporal insertions = plain insertions (insertion still checks for edge existance).
        duration = cppexp::benchmark(do_insert);
    } else if (insertion_routine == "temporal_deletions") {
        auto collected = collect_random_edges();
        duration = cppexp::benchmark(do_temporal_deletions, collected);
    } else if (insertion_routine == "bfs") {
        size_t num_iters = 100;
        size_t num_inserts = 100000;
        size_t off = 0;

        for (size_t i = 0; i < edges.size(); ++i) {
            if (off + num_iters * num_inserts >= edges.size())
                break;
            graph.add_edge(edges[off].source, edges[off].target.vertex, 1);
            ++off;
        }

        duration = cppexp::benchmark([&] {
            for (size_t l = 0; l < 100; ++l) {
                for (size_t i = 0; i < num_inserts; ++i) {
                    if (off >= edges.size())
                        break;
                    graph.add_edge(edges[off].source, edges[off].target.vertex, 1);
                    ++off;
                }

                std::uniform_int_distribution<dhb::Vertex> bfs_distrib(0, edges.size() - 1);
                auto u = edges[bfs_distrib(prng)].source;
                auto parents = BFS_with_edge_map(graph, u);
                free(parents);
            }
            return true;
        });
    } else if (insertion_routine == "spgemm") {
        std::vector<size_t> row_ptrs(as_nv + 1);
        std::vector<dhb::Vertex> col_inds(edges.size());
        std::vector<double> vals(edges.size());
        std::sort(edges.begin(), edges.end(), [](auto l, auto r) { return l.source < r.source; });
        size_t l = 0;
        row_ptrs.push_back(0);
        for (size_t i = 0; i < as_nv; ++i) {
            while (edges[l].source == i) {
                col_inds[l] = edges[l].target.vertex;
                vals[l] = edges[l].target.data.weight;
                ++l;
            }
            row_ptrs[i + 1] = l;
        }
        if (l != edges.size())
            throw std::runtime_error("corrupted CSR");

        auto for_row = [&](dhb::Vertex i, auto f) {
            for (size_t p = row_ptrs[i]; p != row_ptrs[i + 1]; ++p)
                f(col_inds[p], vals[p]);
        };

        duration = cppexp::benchmark([&] {
            size_t nnz = 0; // Stop after a number of non-zeros.
            std::vector<dhb::Vertex> marker(as_nv);
            for (dhb::Vertex i = 0; i < as_nv; ++i) {
                if (nnz >= 100'000'000)
                    break;
                for_row(i, [&](dhb::Vertex j, double wl) {
                    for_row(j, [&](dhb::Vertex k, double wr) {
                        if (marker[k] != i) {
                            marker[k] = i;
                            graph.add_edge(i, k, wl * wr);
                            ++nnz;
                        }
                    });
                });
            }
            return true;
        });
    } else {
        std::cerr << "Insertion routine [" << insertion_routine << "] unknown!" << std::endl;
        return -7;
    }

    duration += std::chrono::duration_cast<std::chrono::milliseconds>(after_ctor - before_ctor);

    size_t vertex_count_after = graph.get_num_vertices();
    size_t edge_count_after = graph.get_num_edges();

    unsigned long long memory_footprint_kb_after_insertion = cppexp::memory_usage_in_kb();

    cppexp::out("experiment", experiment_name);
    cppexp::out("graph", graph_path.filename());
    cppexp::out("edges_read_in", edges_read_in);
    cppexp::out("batch_size", batch_size);
    bool temporal_graph = false;
    cppexp::out("temporal_graph", temporal_graph);
    cppexp::out("insertion_routine", insertion_routine);
    unsigned int insert_factor = 15;
    cppexp::out("insert_factor", insert_factor);
    cppexp::out("format", "terrace");
    cppexp::out("vertex_count_before", vertex_count_before);
    cppexp::out("edge_count_before", edge_count_before);
    cppexp::out("edge_count_after", edge_count_after);
    cppexp::out("duration_ms", duration.count());
    cppexp::out("memory_footprint_kb_before_construction", memory_footprint_kb_before_construction);
    cppexp::out("memory_footprint_kb_after_construction", memory_footprint_kb_after_construction);
    cppexp::out("memory_footprint_kb_after_insertion", memory_footprint_kb_after_insertion);
    bool update = false;
    cppexp::out("update", update);
    cppexp::out("vertex_count_after", vertex_count_after);

    return 0;
}
