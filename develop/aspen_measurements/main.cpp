#include "lib/CLI11.hpp"

#include <dhb/graph.h>

#include <gdsb/batcher.h>
#include <gdsb/experiment.h>
#include <gdsb/graph.h>
#include <gdsb/graph_io.h>
#include <gdsb/timer.h>

// clang-format off
#include <graph/api.h>
#include <algorithms/BFS.h>
// clang-format on

#include <omp.h>

#include <cassert>
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"aspen_measurements"};

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

    gdsb::out("threads", num_workers());
    if (desired_thread_count && num_workers() != desired_thread_count)
        throw std::runtime_error("number of workers does not match desired thread count");
    bool is_parallel = num_workers() > 1;

    std::mt19937 prng{42};
    dhb::Edges edges;
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    std::ifstream graph_input(graph_path);
    gdsb::TimestampedEdges<dhb::Edges> timestamped_edges;

    size_t edges_read_in = 0;
    size_t as_nv = 0;
    if (!temporal_graph_file) {
        gdsb::read_graph_unweighted<dhb::Vertex>(graph_input,
                                                   [&](unsigned int u, unsigned int v) {
                                                       auto w = distrib(prng);
                                                       // Generate a random edge weight in [a, b).
                                                       edges.push_back({u, {v, w, 0u}});
                                                   });
        edges_read_in = edges.size();
        as_nv = dhb::graph::vertex_count(edges);
    } else {
        gdsb::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int t) {
                auto w = distrib(prng);
                // Generate a random edge weight in [a, b).
                timestamped_edges.edges.push_back(dhb::Edge(u, {v, {w, 0u}}));
                timestamped_edges.timestamps.push_back(t);
            });

        timestamped_edges = gdsb::sort(timestamped_edges);
        edges = timestamped_edges.edges;
    }

    edges_read_in = edges.size();
    as_nv = dhb::graph::vertex_count(edges);

    std::chrono::milliseconds duration;

    unsigned long long memory_footprint_kb_before_construction = gdsb::memory_usage_in_kb();
    auto VG = empty_treeplus_graph();
    unsigned long long memory_footprint_kb_after_construction = gdsb::memory_usage_in_kb();

    auto S = VG.acquire_version();
    size_t vertex_count_before = S.graph.num_vertices();
    size_t edge_count_before = S.graph.num_edges();
    VG.release_version(std::move(S));

    if (insertion_routine == "temporal_deletions") {
        auto updates = pbbs::sequence<tuple<uintV, uintV>>(edges.size());
        for (size_t i = 0; i < edges.size(); ++i)
            updates[i] = make_pair(edges[i].source, edges[i].target.vertex);
        //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
        VG.insert_edges_batch(edges.size(), updates.begin(), false, true, as_nv, true);
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

    auto do_insert = [&]() {
        auto updates = pbbs::sequence<tuple<uintV, uintV>>(1);
        for (dhb::Edge const& e : edges) {
            updates[0] = make_pair(e.source, e.target.vertex);
            //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
            VG.insert_edges_batch(1, updates.begin(), false, true, as_nv, !is_parallel);
        }
        return true;
    };

    auto do_insert_bulk = [&]() {
        auto updates = pbbs::sequence<tuple<uintV, uintV>>(edges.size());
        for (size_t i = 0; i < edges.size(); ++i)
            updates[i] = make_pair(edges[i].source, edges[i].target.vertex);
        //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
        VG.insert_edges_batch(edges.size(), updates.begin(), false, true, as_nv, !is_parallel);
        return true;
    };

    auto do_insert_batch = [&]() {
        if (!batch_size)
            throw std::runtime_error("batch size required");
        gdsb::Batcher<dhb::Edges> batcher(std::begin(edges), std::end(edges), batch_size);

        gdsb::Batch<dhb::Edges> batch = batcher.next(1);
        auto updates = pbbs::sequence<tuple<uintV, uintV>>(batch_size);
        while (batch.begin != batcher.end()) {
            size_t i = 0;
            for (auto it = batch.begin; it != batch.end; ++it)
                updates[i++] = make_tuple(it->source, it->target.vertex);
            //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
            VG.insert_edges_batch(i, updates.begin(), false, true, as_nv, !is_parallel);
            batch = batcher.next(1);
        }
        return true;
    };

    auto do_temporal_deletions = [&](struct versioned_graph<treeplus_graph>& VG,
                                     dhb::Edges const& collected) {
        auto updates = pbbs::sequence<tuple<uintV, uintV>>(1);
        for (dhb::Edge const& e : edges) {
            updates[0] = make_pair(e.source, e.target.vertex);
            //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
            VG.delete_edges_batch(1, updates.begin(), false, true, as_nv, !is_parallel);
        }
        return true;
    };

    if (insertion_routine == "insert") {
        duration = gdsb::benchmark(do_insert);
    } else if (insertion_routine == "insert_bulk") {
        duration = gdsb::benchmark(do_insert_bulk);
    } else if (insertion_routine == "insert_batch") {
        duration = gdsb::benchmark(do_insert_batch);
    } else if (insertion_routine == "temporal_insertions") {
        // For Aspen, temporal insertions equal plain insertions (since there are no updates).
        duration = gdsb::benchmark(do_insert);
    } else if (insertion_routine == "temporal_deletions") {
        auto collected = collect_random_edges();
        duration = gdsb::benchmark(do_temporal_deletions, VG, collected);
    } else if (insertion_routine == "bfs") {
        size_t num_iters = 100;
        size_t num_inserts = 100000;
        size_t off = 0;
        size_t i;

        auto initial = pbbs::sequence<tuple<uintV, uintV>>(edges.size());
        for (i = 0; i < edges.size(); ++i) {
            if (off + num_iters * num_inserts >= edges.size())
                break;
            initial[i] = make_pair(edges[off].source, edges[off].target.vertex);
            ++off;
        }
        //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
        VG.insert_edges_batch(i, initial.begin(), false, true, as_nv, !is_parallel);

        // Add self-loops to ensure that all vertex IDs are mapped
        // (Aspen segfaults if they are not).
        auto loops = pbbs::sequence<tuple<uintV, uintV>>(as_nv);
        for (i = 0; i < as_nv; ++i)
            loops[i] = make_pair(i, i);
        VG.insert_edges_batch(i, loops.begin(), false, true, as_nv, !is_parallel);

        duration = gdsb::benchmark([&] {
            for (size_t l = 0; l < 100; ++l) {
                auto updates = pbbs::sequence<tuple<uintV, uintV>>(num_inserts);
                for (i = 0; i < num_inserts; ++i) {
                    if (off >= edges.size())
                        break;
                    updates[i] = make_pair(edges[off].source, edges[off].target.vertex);
                    ++off;
                }
                //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
                VG.insert_edges_batch(i, updates.begin(), false, true, as_nv, !is_parallel);

                auto s_bfs = VG.acquire_version();
                auto vc = s_bfs.graph.num_vertices();
                std::uniform_int_distribution<dhb::Vertex> bfs_distrib(0, edges.size() - 1);
                auto u = edges[bfs_distrib(prng)].source;
                // BFS_Fetch(s_bfs.graph, u);
                BFS(s_bfs.graph, u);
                VG.release_version(std::move(s_bfs));
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

        duration = gdsb::benchmark([&] {
            size_t nnz = 0; // Stop after a number of non-zeros.
            std::vector<dhb::Vertex> marker(as_nv);
            std::vector<tuple<uintV, uintV>> stack;
            for (dhb::Vertex i = 0; i < as_nv; ++i) {
                stack.clear();
                if (nnz >= 100'000'000)
                    break;
                for_row(i, [&](dhb::Vertex j, double) {
                    for_row(j, [&](dhb::Vertex k, double) {
                        if (marker[k] != i) {
                            marker[k] = i;
                            stack.push_back({i, k});
                            ++nnz;
                        }
                    });
                });
                if (!stack.size())
                    continue;
                //(/*sorted=*/, /*remove_dups=*/, /*n*/, /*run_seq=*/)
                VG.insert_edges_batch(stack.size(), stack.data(), false, true, as_nv, true);
            }
            return true;
        });
    } else {
        std::cerr << "Insertion routine [" << insertion_routine << "] unknown!" << std::endl;
        return -7;
    }

    auto S2 = VG.acquire_version();
    size_t vertex_count_after = S2.graph.num_vertices();

    unsigned long long memory_footprint_kb_after_insertion = gdsb::memory_usage_in_kb();

    size_t edge_count_after = S2.graph.num_edges();
    VG.release_version(std::move(S2));

    gdsb::out("experiment", experiment_name);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("edges_read_in", edges_read_in);
    gdsb::out("batch_size", batch_size);
    bool temporal_graph = false;
    gdsb::out("temporal_graph", temporal_graph);
    gdsb::out("insertion_routine", insertion_routine);
    unsigned int insert_factor = 15;
    gdsb::out("insert_factor", insert_factor);
    gdsb::out("format", "aspen");
    gdsb::out("vertex_count_before", vertex_count_before);
    gdsb::out("edge_count_before", edge_count_before);
    gdsb::out("edge_count_after", edge_count_after);
    gdsb::out("duration_ms", duration.count());
    gdsb::out("memory_footprint_kb_before_construction", memory_footprint_kb_before_construction);
    gdsb::out("memory_footprint_kb_after_construction", memory_footprint_kb_after_construction);
    gdsb::out("memory_footprint_kb_after_insertion", memory_footprint_kb_after_insertion);
    bool update = false;
    gdsb::out("update", update);
    gdsb::out("vertex_count_after", vertex_count_after);

    return 0;
}
