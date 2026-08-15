#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

static int free_port() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    const int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(sock, (struct sockaddr *) &addr, sizeof(addr));
    socklen_t len = sizeof(addr);
    getsockname(sock, (struct sockaddr *) &addr, &len);
    const int port = ntohs(addr.sin_port);
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    return port;
}

int main() {
    ggml_backend_load_all();

    // two in-process RPC servers on free ports
    const std::string endpoint_a = "127.0.0.1:" + std::to_string(free_port());
    const std::string endpoint_b = "127.0.0.1:" + std::to_string(free_port());

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    std::thread th_a([&]() { ggml_backend_rpc_start_server(endpoint_a.c_str(), nullptr, 1, 1, &cpu_dev); });
    std::thread th_b([&]() { ggml_backend_rpc_start_server(endpoint_b.c_str(), nullptr, 1, 1, &cpu_dev); });
    th_a.detach();
    th_b.detach();

    // wait for both servers to accept connections
    bool ready = false;
    for (int i = 0; i < 100 && !ready; i++) {
        size_t free_mem, total_mem;
        ggml_backend_rpc_get_device_memory(endpoint_a.c_str(), 0, &free_mem, &total_mem);
        ready = total_mem > 0;
        ggml_backend_rpc_get_device_memory(endpoint_b.c_str(), 0, &free_mem, &total_mem);
        ready = ready && total_mem > 0;
        if (!ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    GGML_ASSERT(ready && "RPC servers did not start");

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a.c_str(), 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b.c_str(), 0);
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    // a lives on server A, w on server B (statically allocated, weights)
    ggml_context * ctx_a = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * a = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, 4);
    ggml_set_name(a, "a");
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, ggml_backend_rpc_buffer_type(endpoint_a.c_str(), 0));
    ggml_backend_buffer_set_usage(buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_context * ctx_w = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, 4);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_rpc_buffer_type(endpoint_b.c_str(), 0));
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // v and y are allocated by the scheduler
    ggml_context * ctx_g = ggml_init({ 2*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true });
    ggml_tensor * v = ggml_view_1d(ctx_g, a, 4, 0); // backend follows a (server A)
    ggml_tensor * y = ggml_mul_mat(ctx_g, w, v);    // backend follows w (server B)
    ggml_cgraph * graph = ggml_new_graph_custom(ctx_g, 8, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_t backends[3] = { backend_a, backend_b, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // a = [1,2,3,4] on A, w = identity on B: y = w @ view(a) == a
    const float a_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));
    const float w_data[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    ggml_backend_tensor_set(w, w_data, 0, sizeof(w_data));

    // before the fix this crashes server B with "invalid data ptr"
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    // verify y == a
    float y_data[4] = { 0.0f };
    ggml_backend_tensor_get(y, y_data, 0, sizeof(y_data));
    for (int i = 0; i < 4; i++) {
        GGML_ASSERT(y_data[i] == a_data[i]);
    }

    printf("PASS: graph with cross-server view computed correctly\n");
    return 0;
}
