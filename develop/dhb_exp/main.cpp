#include "lib/CLI11.hpp"

#include <dhb/dynamic_hashed_blocks.h>
#include <dhb/graph.h>

#include <gdsb/batcher.h>
#include <gdsb/experiment.h>
#include <gdsb/graph.h>
#include <gdsb/graph_io.h>
#include <gdsb/timer.h>

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::experimental::filesystem;

enum class InsertionRoutine {
    random,
    insert,
    insert_bulk,
    insert_batch,
    temporal_insertions,
    temporal_updates,
    temporal_deletions,
    bfs,
    spgemm,
    unknown
};

InsertionRoutine get(std::string insertion_routine) {
    if (insertion_routine == "rand_insert") {
        return InsertionRoutine::random;
    } else if (insertion_routine == "insert") {
        return InsertionRoutine::insert;
    } else if (insertion_routine == "insert_bulk") {
        return InsertionRoutine::insert_bulk;
    } else if (insertion_routine == "insert_batch") {
        return InsertionRoutine::insert_batch;
    } else if (insertion_routine == "temporal_insertions") {
        return InsertionRoutine::temporal_insertions;
    } else if (insertion_routine == "temporal_updates") {
        return InsertionRoutine::temporal_updates;
    } else if (insertion_routine == "temporal_deletions") {
        return InsertionRoutine::temporal_deletions;
    } else if (insertion_routine == "bfs") {
        return InsertionRoutine::bfs;
    } else if (insertion_routine == "spgemm") {
        return InsertionRoutine::spgemm;
    } else {
        return InsertionRoutine::unknown;
    }
}

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

dhb::Edges retrieve_edges(InsertionRoutine insertion_routine, fs::path const& graph_path,
                          bool const temporal_graph, bool const weighted_graph_file,
                          std::mt19937& prng, unsigned int const vertex_count,
                          unsigned int const insert_factor) {
    if (insertion_routine == InsertionRoutine::random) {
        return create_random_edges(vertex_count * insert_factor, vertex_count - 1, prng);
    }

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
        gdsb::read_graph_unweighted<dhb::Vertex>(
            graph_input, [&](unsigned int u, unsigned int v) {
                edges.push_back(dhb::Edge(u, {v, {distrib(prng), 0u}}));
            });
    }

    return edges;
}

void unknown_insertion_routine(std::string routine) {
    std::cerr << "Insertion routine [" << routine << "] unknown!";
    std::exit(EXIT_FAILURE);
}

dhb::Edges collect_random_edges(dhb::Edges const& edges, std::mt19937& prng) {
    size_t const collected_edges_count = edges.size() / 2;
    std::vector<dhb::Edge> collected(collected_edges_count);
    collected.reserve(collected_edges_count);

    std::uniform_int_distribution<size_t> edge_distrib(0, edges.size() - 1);
    std::uniform_real_distribution<float> w_distrib{0.0f, 1.0f};
    for (size_t i = 0; i < collected_edges_count; ++i) {
        auto edge = edges[edge_distrib(prng)];
        edge.target.data.weight = w_distrib(prng);
        collected.emplace_back(edge);
    }

    return collected;
}

int main(int argc, char** argv) {
    // If you don't know CLI11 yet, you'll find the doc on
    // https://github.com/CLIUtils/CLI11/blob/master/README.md
    CLI::App app{"dhb_exp"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");
    std::string graph = "";
    app.add_option("-g,--graph", graph, "Full path to graph file.");
    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");
    std::string insertion_routine_input{"none"};
    app.add_option("-r,--insertion-routine", insertion_routine_input,
                   "Insertion routine to be used.");
    unsigned int insert_factor = 15;
    app.add_option("-i,--insert-factor", insert_factor,
                   "Factor of random insertions depending on the count of vertices.");
    unsigned int omp_thread_count = 1;
    app.add_option("-t,--thread-count", omp_thread_count,
                   "# of threads to execute OMP regions with.");
    size_t batch_size = 0;
    app.add_option("-b, --batch-size", batch_size,
                   "Size of batch for inserting edges of graph. If set to 0 all edges will be "
                   "inserted one batch.");
    bool temporal_graph = false;
    app.add_flag("-y,--temporal-graph", temporal_graph,
                 "Flag in case the graph file contains timestamps."
                 "This will sort all edges based on the time stamp.");
    bool update = false;
    app.add_flag(
        "-u,--update", update,
        "Perform updates: this will check first if edge {u, v} in E and updates the weight.");
    dhb::Vertex vertex_count = 0;
    app.add_option("-n,--vertex-count", vertex_count, "Defines the count of vertices.");

    CLI11_PARSE(app, argc, argv);

    InsertionRoutine insertion_routine = get(insertion_routine_input);
    if (insertion_routine == InsertionRoutine::unknown) {
        unknown_insertion_routine(insertion_routine_input);
    }

    fs::path graph_path(std::move(graph));
    if (!fs::exists(graph_path) && insertion_routine != InsertionRoutine::random) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    omp_set_num_threads(omp_thread_count);
    assert(omp_thread_count == omp_get_max_threads());

    std::mt19937 prng{42};

    dhb::Edges edges = retrieve_edges(insertion_routine, graph_path, temporal_graph,
                                      weighted_graph_file, prng, vertex_count, insert_factor);

    unsigned int const edges_read_in = edges.size();

    unsigned long long const memory_footprint_kb_before_construction = gdsb::memory_usage_in_kb();
    dhb::Matrix<dhb::Weight> matrix(dhb::graph::vertex_count(edges));
    switch (insertion_routine) {
    case InsertionRoutine::random:              // intentional fall through
    case InsertionRoutine::insert:              // intentional fall through
    case InsertionRoutine::insert_bulk:         // intentional fall through
    case InsertionRoutine::insert_batch:        // intentional fall through
    case InsertionRoutine::temporal_insertions: // intentional fall through
    case InsertionRoutine::bfs:                 // intentional fall through
    case InsertionRoutine::spgemm:
        break;
    case InsertionRoutine::temporal_updates:   // intentional fall through
    case InsertionRoutine::temporal_deletions: // intentional fall through
        for (const auto& e : edges)
            matrix.neighbors(e.source).insert(e.target.vertex, e.target.data.weight);
        break;
    default:
        unknown_insertion_routine(insertion_routine_input);
    }

    vertex_count = matrix.vertices_count();
    unsigned long long const memory_footprint_kb_after_construction = gdsb::memory_usage_in_kb();
    unsigned int const max_degree_before = std::get<0>(dhb::max_degree(matrix));
    dhb::Vertex const max_degree_node_before = std::get<1>(dhb::max_degree(matrix));
    unsigned int const edge_count_before = matrix.edges_count();

    auto dhb_key = [](dhb::Edge e) { return e.source; };

    auto insert_f_no_update = [&](dhb::Edge e) {
        matrix.neighbors(e.source).insert(e.target.vertex, e.target.data.weight);
    };
    auto insert_f_with_update = [&](dhb::Edge e) {
        std::tuple<dhb::BlockState<dhb::Weight>::iterator, bool> insertion_result =
            matrix.neighbors(e.source).insert(e.target.vertex, e.target.data.weight);
        if (!std::get<1>(insertion_result)) {
            std::get<0>(insertion_result)->data() = e.target.data.weight;
        }
    };
    dhb::BatchParallelizer<dhb::Edge> parallelizer;

    auto do_insert = [&]() {
        if (batch_size)
            throw std::runtime_error("batches are not supported");
        if (omp_thread_count != 1)
            throw std::runtime_error("parallelism is not supported");
        for (dhb::Edge e : edges)
            insert_f_no_update(e);
        return true;
    };

    auto do_insert_bulk = [&]() {
        if (batch_size)
            throw std::runtime_error("batches are not supported");
        std::vector<unsigned int> degrees(vertex_count);
        if (omp_thread_count > 1) {
#pragma omp parallel
            {
                parallelizer.distribute(edges.begin(), edges.end(), dhb_key);

                // First, compute degrees.
                parallelizer.map(edges.begin(), edges.end(),
                                 [&](const auto& e) { ++degrees[e.source]; });

                // Now insert all edges.
                parallelizer.map(edges.begin(), edges.end(), [&](const auto& e) {
                    // Preallocate memory when we see the source for the first time.
                    if (degrees[e.source]) {
                        matrix.preallocate(e.source, degrees[e.source]);
                        degrees[e.source] = 0;
                    }
                    insert_f_no_update(e);
                });
            }
        } else {
            for (const auto& e : edges)
                ++degrees[e.source];
            for (size_t i = 0; i < vertex_count; ++i)
                matrix.preallocate(i, degrees[i]);
            for (const auto& e : edges)
                insert_f_no_update(e);
        }
        return true;
    };

    auto do_insert_batch = [&]() {
        if (!batch_size)
            throw std::runtime_error("batch size required");
        gdsb::Batcher<dhb::Edges> batcher(std::begin(edges), std::end(edges), batch_size);

        gdsb::Batch<dhb::Edges> batch = batcher.next(omp_thread_count);
        std::vector<unsigned int> new_degrees(vertex_count);
        std::vector<unsigned int> done(batch_size);
        size_t c = 1;
        while (batch.begin != batcher.end()) {
            if (omp_thread_count > 1) {
#pragma omp parallel
                {
                    parallelizer.distribute(dhb::Edges::const_iterator(batch.begin), batch.end,
                                            dhb_key);

                    // First, compute degrees.
                    parallelizer.map(dhb::Edges::const_iterator(batch.begin), batch.end,
                                     [&](auto& e) {
                                         auto adj = matrix.neighbors(e.source);
                                         auto it = adj.iterator_to(e.target.vertex);
                                         if (it != adj.end()) {
                                             e.source = dhb::invalidVertex();
                                         } else {
                                             ++new_degrees[e.source];
                                         }
                                     });

                    // Now insert all edges.
                    parallelizer.map(
                        dhb::Edges::const_iterator(batch.begin), batch.end, [&](const auto& e) {
                            if (e.source == dhb::invalidVertex())
                                return;
                            // Preallocate memory when we see the source for the first time.
                            if (new_degrees[e.source]) {
                                matrix.preallocate(e.source, matrix.neighbors(e.source).degree() +
                                                                 new_degrees[e.source]);
                                new_degrees[e.source] = 0;
                            }
                            insert_f_no_update(e);
                        });
                }
            } else {
                for (auto bit = batch.begin; bit != batch.end; ++bit) {
                    const auto& e = *bit;
                    auto idx = &e - &(*batch.begin);

                    auto adj = matrix.neighbors(e.source);
                    auto it = adj.iterator_to(e.target.vertex);
                    if (it != adj.end()) {
                        done[idx] = c;
                    } else {
                        ++new_degrees[e.source];
                    }
                }
                for (auto bit = batch.begin; bit != batch.end; ++bit) {
                    const auto& e = *bit;
                    auto idx = &e - &(*batch.begin);

                    if (done[idx] == c)
                        continue;
                    if (new_degrees[e.source]) {
                        matrix.preallocate(e.source, matrix.neighbors(e.source).degree() +
                                                         new_degrees[e.source]);
                        new_degrees[e.source] = 0;
                    }
                    insert_f_no_update(e);
                }
            }
            batch = batcher.next(omp_thread_count);
            ++c;
        }
        return true;
    };

    auto do_temporal_insertions = [&]() {
        if (batch_size)
            throw std::runtime_error("batches are not supported");
        if (omp_thread_count != 1)
            throw std::runtime_error("parallelism is not supported");
        for (dhb::Edge e : edges)
            insert_f_with_update(e);
        return true;
    };

    auto do_temporal_updates = [&](dhb::Edges& collected) {
        if (batch_size)
            throw std::runtime_error("batches are not supported");
        if (omp_thread_count != 1)
            throw std::runtime_error("parallelism is not supported");
        for (dhb::Edge e : collected)
            insert_f_with_update(e);
        return true;
    };

    auto do_temporal_deletions = [&](dhb::Edges& collected) {
        if (batch_size)
            throw std::runtime_error("batches are not supported");
        if (omp_thread_count != 1)
            throw std::runtime_error("parallelism is not supported");
        for (auto& edge : collected)
            matrix.removeEdge(edge.source, edge.target.vertex);
        return true;
    };

    std::chrono::milliseconds duration = [&]() {
        switch (insertion_routine) {
        case InsertionRoutine::random:
            if (batch_size > 0) {
                return gdsb::benchmark(do_insert_batch);
            } else {
                return gdsb::benchmark(do_insert);
            }
        case InsertionRoutine::insert:
            return gdsb::benchmark(do_insert);
        case InsertionRoutine::insert_bulk:
            return gdsb::benchmark(do_insert_bulk);
        case InsertionRoutine::insert_batch:
            return gdsb::benchmark(do_insert_batch);
        case InsertionRoutine::temporal_insertions:
            return gdsb::benchmark(do_temporal_insertions);
        case InsertionRoutine::temporal_updates: {
            auto collected = collect_random_edges(edges, prng);
            return gdsb::benchmark(do_temporal_updates, collected);
        }
        case InsertionRoutine::temporal_deletions: {
            auto collected = collect_random_edges(edges, prng);
            return gdsb::benchmark(do_temporal_deletions, collected);
        }
        case InsertionRoutine::bfs: {
            size_t num_iters = 100;
            size_t num_inserts = 100000;
            size_t off = 0;

            for (size_t i = 0; i < edges.size(); ++i) {
                if (off + num_iters * num_inserts >= edges.size())
                    break;
                insert_f_no_update(edges[off]);
                ++off;
            }

            // Same approach as NetworKit::Traversal::BFSfrom.
            auto matrixBFSfrom = [&](dhb::Vertex r, auto f) {
                std::vector<bool> marked(vertex_count);
                std::queue<dhb::Vertex> q, qNext;
                uint64_t dist = 0;
                // enqueue start nodes
                q.push(r);
                marked[r] = true;
                do {
                    const auto u = q.front();
                    q.pop();
                    // apply function
                    f(u, dist);
                    for (const auto& ent : matrix.neighbors(u)) {
                        auto v = ent.vertex();
                        if (!marked[v]) {
                            qNext.push(v);
                            marked[v] = true;
                        }
                    };
                    if (q.empty() && !qNext.empty()) {
                        q.swap(qNext);
                        ++dist;
                    }
                } while (!q.empty());
            };

            return gdsb::benchmark([&] {
                uint64_t sum_of_dists = 0;
                std::uniform_int_distribution<dhb::Vertex> bfs_distrib(0, vertex_count - 1);
                for (size_t l = 0; l < num_iters; ++l) {
                    for (size_t i = 0; i < num_inserts; ++i) {
                        if (off >= edges.size())
                            break;
                        insert_f_no_update(edges[off]);
                        ++off;
                    }

                    auto u = bfs_distrib(prng);
                    matrixBFSfrom(u, [&](auto, auto dist) { sum_of_dists += dist; });
                }
                gdsb::out("sum_of_dists", sum_of_dists);
                return true;
            });
        }
        case InsertionRoutine::spgemm: {
            std::vector<size_t> row_ptrs(vertex_count + 1);
            std::vector<dhb::Vertex> col_inds(edges.size());
            std::vector<double> vals(edges.size());
            std::sort(edges.begin(), edges.end(),
                      [](auto l, auto r) { return l.source < r.source; });
            size_t l = 0;
            row_ptrs.push_back(0);
            for (size_t i = 0; i < vertex_count; ++i) {
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

            return gdsb::benchmark([&] {
                size_t nnz = 0; // Stop after a number of non-zeros.
                for (dhb::Vertex i = 0; i < vertex_count; ++i) {
                    if (nnz >= 100'000'000)
                        break;
                    for_row(i, [&](dhb::Vertex j, double l_val) {
                        for_row(j, [&](dhb::Vertex k, double r_val) {
                            auto prod = l_val * r_val;
                            auto res = matrix.neighbors(i).insert(k, prod);
                            if (std::get<1>(res))
                                ++nnz;
                            if (!std::get<1>(res))
                                std::get<0>(res)->data() += prod;
                        });
                    });
                }
                return true;
            });
        }
        default:
            unknown_insertion_routine(insertion_routine_input);
            return std::chrono::milliseconds(0);
        }
    }();

    unsigned long long const memory_footprint_kb_after_insertion = gdsb::memory_usage_in_kb();
    unsigned int const max_degree_after = std::get<0>(max_degree(matrix));
    dhb::Vertex const max_degree_node_after = std::get<1>(max_degree(matrix));
    unsigned int const edge_count_after = matrix.edges_count();

    gdsb::out("experiment", experiment_name);
    gdsb::out("threads", omp_thread_count);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("edges_read_in", edges_read_in);
    gdsb::out("batch_size", batch_size);
    gdsb::out("temporal_graph", temporal_graph);
    gdsb::out("insertion_routine", insertion_routine_input);
    gdsb::out("insert_factor", insert_factor);
    gdsb::out("format", "DHB");
    gdsb::out("vertex_count", vertex_count);
    gdsb::out("edge_count_before", edge_count_before);
    gdsb::out("edge_count_after", edge_count_after);
    gdsb::out("max_degree_before", max_degree_before);
    gdsb::out("max_degree_after", max_degree_after);
    gdsb::out("max_degree_node_before", max_degree_node_before);
    gdsb::out("max_degree_node_after", max_degree_node_after);
    gdsb::out("duration_ms", duration.count());
    gdsb::out("memory_footprint_kb_before_construction", memory_footprint_kb_before_construction);
    gdsb::out("memory_footprint_kb_after_construction", memory_footprint_kb_after_construction);
    gdsb::out("memory_footprint_kb_after_insertion", memory_footprint_kb_after_insertion);
    gdsb::out("update", update);

    return 0;
}
