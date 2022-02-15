#include <dhb/graph.h>

#include <gdsb/batcher.h>
#include <gdsb/experiment.h>
#include <gdsb/graph_io.h>

#include "lib/CLI11.hpp"

#include <stinger_core/stinger.h>
#include <stinger_core/stinger_traversal.h>
#include <stinger_core/xmalloc.h>
#include <stinger_utils/stinger_utils.h>

#include <omp.h>

#include <cassert>
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

// STINGER makes use of "restrict" in macros w/o defining it.
#define restrict

int main(int argc, char** argv) {
    CLI::App app{"Stinger_measurements"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");

    std::string graph_raw_path = "graphs/foodweb-baydry.konect";
    app.add_option("-g,--graph", graph_raw_path, "Full path to graph file.");

    std::string insertion_routine{"none"};
    app.add_option("-r,--insertion-routine", insertion_routine, "Insertion routine to be used.");

    unsigned int omp_thread_count = 1;
    app.add_option("-t,--thread-count", omp_thread_count,
                   "# of threads to execute OMP regions with.");

    size_t batch_size = 0;
    app.add_option("-b, --batch-size", batch_size,
                   "Size of batch for inserting edges of graph. If set to 0 all edges will be "
                   "inserted one batch.");

    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");

    unsigned int insert_factor = 15;
    app.add_option("-i,--insert-factor", insert_factor,
                   "Factor of random insertions depending on the count of vertices.");

    unsigned int vertex_count = 0;
    app.add_option("-n,--vertex-count", vertex_count, "Defines the count of vertices.");

    bool temporal_graph_file = false;
    app.add_flag("-y,--temporal-graph", temporal_graph_file,
                 "Flag in case the graph file contains timestamps."
                 "This will sort all edges based on the time stamp.");

    CLI11_PARSE(app, argc, argv);

    namespace fs = std::experimental::filesystem;
    fs::path graph_path(std::move(graph_raw_path));

    if (!fs::exists(graph_path)) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        return -1;
    }

    omp_set_num_threads(omp_thread_count);
    assert(omp_thread_count == omp_get_max_threads());
    gdsb::out("threads", omp_get_max_threads());

    std::mt19937 prng{42};
    dhb::Edges edges;
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    std::ifstream graph_input(graph_path);
    gdsb::TimestampedEdges<dhb::Edges> timestamped_edges;

    struct stinger_config_t* stinger_config =
        (struct stinger_config_t*)xcalloc(1, sizeof(struct stinger_config_t));
    size_t edges_read_in = 0;

    if (temporal_graph_file) {
        gdsb::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int t) {
                auto w = distrib(prng);
                // Generate a random edge weight in [a, b).
                timestamped_edges.edges.push_back({u, {v, w, 0u}});
                timestamped_edges.timestamps.push_back(t);
            });

        timestamped_edges = gdsb::sort(timestamped_edges);
        edges = std::move(timestamped_edges.edges);
    } else {
        gdsb::read_graph_unweighted<dhb::Vertex>(graph_input,
                                                   [&](unsigned int u, unsigned int v) {
                                                       auto w = distrib(prng);
                                                       // Generate a random edge weight in [a, b).
                                                       edges.push_back({u, {v, w, 0u}});
                                                   });
        edges_read_in = edges.size();
        stinger_config->nv = dhb::graph::vertex_count(edges);
    }

    edges_read_in = edges.size();
    stinger_config->nv = dhb::graph::vertex_count(edges);

    std::chrono::milliseconds duration;

    unsigned long long memory_footprint_kb_before_construction = gdsb::memory_usage_in_kb();

    // STINGER_DEFAULT_NEB_FACTOR is set to 4 just as we did for bsize
    stinger_config->nebs = STINGER_DEFAULT_NEB_FACTOR * stinger_config->nv;
    // only one edge type
    stinger_config->netypes = 1;
    // only one vertex type
    stinger_config->nvtypes = 1;

    // For the memory size I'm not sure in which unit they expect it.
    // I assume here based on the source code it's actually in bytes
    // stinger_config->memory_size = 8000000000;
    //
    // Update: I think we should not set the memory size parameter as it will be set
    // to half of the available memory on the system in with getMemorySize() in getMemorySize.c
    // But on the other hand perhaps we want to set it to something greater, then we better
    // $ export STINGER_MAX_MEMSIZE=8g
    // .. for 8 GB.
    stinger_config->no_map_none_etype = 0;
    stinger_config->no_map_none_vtype = 0;
    stinger_config->no_resize = 0;
    stinger* graph = stinger_new_full(stinger_config);

    if (0 != stinger_consistency_check(graph, graph->max_nv)) {
        std::cerr << "Stinger DS not consistent!" << std::endl;
        return -1;
    }

    unsigned long long memory_footprint_kb_after_construction = gdsb::memory_usage_in_kb();

    unsigned int edge_count_before = stinger_total_edges(graph);

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

    auto add_all_edges = [&]() {
        int ret = 1;
        for (dhb::Edge const& edge : edges) {
            ret = stinger_insert_edge(graph, 0, edge.source, edge.target.vertex,
                                      static_cast<int64_t>(edge.target.data.weight), 0);
            assert(1 == ret);
        }
        return true;
    };

    auto set_weight_for_all = [&]() {
        int ret = 1;
        for (size_t i = 0; i < edges.size(); ++i) {
            ret = stinger_update_directed_edge(graph, 0, edges[i].source, edges[i].target.vertex,
                                               static_cast<int64_t>(edges[i].target.data.weight), 0,
                                               STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
            assert(1 == ret);
        }
        return true;
    };

    auto set_weight_for_all_bulk = [&]() {
#pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i) {
            int ret =
                stinger_update_directed_edge(graph, 0, edges[i].source, edges[i].target.vertex,
                                             static_cast<int64_t>(edges[i].target.data.weight), 0,
                                             STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
            assert(1 == ret);
        }
        return true;
    };

    auto set_weight_for_all_batch = [&]() {
        if (!batch_size)
            throw std::runtime_error("batch size required");
        gdsb::Batcher<dhb::Edges> batcher(std::begin(edges), std::end(edges), batch_size);

        gdsb::Batch<dhb::Edges> batch = batcher.next(1);
        while (batch.begin != batcher.end()) {
            size_t actual_size = batch.end - batch.begin;
#pragma omp parallel for
            for (size_t i = 0; i < actual_size; ++i) {
                auto it = batch.begin + i;
                int ret =
                    stinger_update_directed_edge(graph, 0, it->source, it->target.vertex,
                                                 static_cast<int64_t>(it->target.data.weight), 0,
                                                 STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
                assert(1 == ret);
            }
            batch = batcher.next(1);
        }
        return true;
    };

    auto update_all = [&](dhb::Edges const& collected) {
        for (size_t i = 0; i < collected.size(); ++i) {
            stinger_update_directed_edge(graph, 0, collected[i].source, collected[i].target.vertex,
                                         static_cast<int64_t>(collected[i].target.data.weight), 0,
                                         STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
        }
        return true;
    };

    auto remove_all = [&](dhb::Edges const& collected) {
        for (size_t i = 0; i < collected.size(); ++i) {
            stinger_remove_edge(graph, 0, collected[i].source, collected[i].target.vertex);
        }
        return true;
    };

    if (insertion_routine == "insert") {
        duration = gdsb::benchmark(add_all_edges);
    } else if (insertion_routine == "insert_bulk") {
        duration = gdsb::benchmark(set_weight_for_all_bulk);
    } else if (insertion_routine == "insert_batch") {
        duration = gdsb::benchmark(set_weight_for_all_batch);
    } else if (insertion_routine == "temporal_insertions") {
        duration = gdsb::benchmark(set_weight_for_all);
    } else if (insertion_routine == "temporal_updates") {
        // Add edges before running benchmark.
        add_all_edges();

        auto collected = collect_random_edges();
        duration = gdsb::benchmark(update_all, collected);
    } else if (insertion_routine == "temporal_deletions") {
        // Add edges before running benchmark.
        add_all_edges();

        auto collected = collect_random_edges();
        duration = gdsb::benchmark(remove_all, collected);
    } else if (insertion_routine == "bfs") {
        size_t num_iters = 100;
        size_t num_inserts = 100000;
        size_t off = 0;

        for (size_t i = 0; i < edges.size(); ++i) {
            if (off + num_iters * num_inserts >= edges.size())
                break;
            int ret =
                stinger_update_directed_edge(graph, 0, edges[off].source, edges[off].target.vertex,
                                             static_cast<int64_t>(edges[off].target.data.weight), 0,
                                             STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
            assert(1 == ret);
            ++off;
        }

        // Same approach as NetworKit::Traversal::BFSfrom.
        auto matrixBFSfrom = [&](dhb::Vertex r, auto f) {
            std::vector<bool> marked(stinger_config->nv);
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
                STINGER_READ_ONLY_FORALL_OUT_EDGES_OF_VTX_BEGIN(graph, u) {
                    auto v = STINGER_RO_EDGE_DEST;
                    if (!marked[v]) {
                        qNext.push(v);
                        marked[v] = true;
                    }
                }
                STINGER_READ_ONLY_FORALL_OUT_EDGES_OF_VTX_END();
                if (q.empty() && !qNext.empty()) {
                    q.swap(qNext);
                    ++dist;
                }
            } while (!q.empty());
        };

        duration = gdsb::benchmark([&] {
            uint64_t sum_of_dists = 0;
            std::uniform_int_distribution<dhb::Vertex> bfs_distrib(0, stinger_config->nv - 1);
            for (size_t l = 0; l < num_iters; ++l) {
                for (size_t i = 0; i < num_inserts; ++i) {
                    if (off >= edges.size())
                        break;
                    int ret = stinger_update_directed_edge(
                        graph, 0, edges[off].source, edges[off].target.vertex,
                        static_cast<int64_t>(edges[off].target.data.weight), 0,
                        STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_SET);
                    ++off;
                }

                auto u = bfs_distrib(prng);
                matrixBFSfrom(u, [&](auto, auto dist) { sum_of_dists += dist; });
            }
            gdsb::out("sum_of_dists", sum_of_dists);
            return true;
        });
    } else if (insertion_routine == "spgemm") {
		std::vector<size_t> row_ptrs(stinger_config->nv + 1);
		std::vector<dhb::Vertex> col_inds(edges.size());
		std::vector<double> vals(edges.size());
		std::sort(edges.begin(), edges.end(),
				  [](auto l, auto r) { return l.source < r.source; });
		size_t l = 0;
		row_ptrs.push_back(0);
		for (size_t i = 0; i < stinger_config->nv; ++i) {
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
			for (dhb::Vertex i = 0; i < stinger_config->nv; ++i) {
				if (nnz >= 100'000'000)
					break;
				for_row(i, [&](dhb::Vertex j, double l_val) {
					for_row(j, [&](dhb::Vertex k, double r_val) {
						auto prod = l_val * r_val;
						int ret =
							stinger_update_directed_edge(graph, 0, i, k, prod, 0,
														 STINGER_EDGE_DIRECTION_OUT, EDGE_WEIGHT_INCR);
						assert(1 == ret);
					});
				});
			}
			return true;
		});
    } else {
        std::cerr << "Insertion routine [" << insertion_routine << "] unknown!" << std::endl;
        return -7;
    }

    unsigned int edge_count_after = stinger_total_edges(graph);
    unsigned long long memory_footprint_kb_after_insertion = gdsb::memory_usage_in_kb();

    // It seems that one vertex is not active here and I assume it's 0
    // this has to do with reading in graphs and what is their ID interval (0, N] or [0, N)
    // TODO: test why this is N-1!
    // Update: using max nv now!
    vertex_count = graph->max_nv;

    gdsb::out("experiment", experiment_name);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("insertion_routine", insertion_routine);
    gdsb::out("edges_read_in", edges_read_in);
    gdsb::out("batch_size", batch_size);
    gdsb::out("format", "Stinger");
    gdsb::out("vertex_count", vertex_count);
    gdsb::out("edge_count_before", edge_count_before);
    gdsb::out("memory_footprint_kb_before_construction", memory_footprint_kb_before_construction);
    gdsb::out("memory_footprint_kb_after_construction", memory_footprint_kb_after_construction);
    gdsb::out("memory_footprint_kb_after_insertion", memory_footprint_kb_after_insertion);
    gdsb::out("edge_count_after", edge_count_after);
    gdsb::out("duration_ms", duration.count());

    xfree(stinger_config);
    stinger_free_all(graph);

    return 0;
}
