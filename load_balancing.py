import os
import time
from functools import wraps

import numpy as np
import psutil


def measure_performance1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        start_time = time.time()
        start_cpu_times = psutil.cpu_times()
        start_mem = process.memory_info().rss
        start_cpu_percent = psutil.cpu_percent(percpu=True)
        start_cpu = psutil.cpu_percent()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_cpu_times = psutil.cpu_times()
        end_mem = process.memory_info().rss
        end_cpu_percent = psutil.cpu_percent(percpu=True)
        end_cpu = psutil.cpu_percent()

        # 计算CPU时间（以秒为单位）
        cpu_time = sum(end_cpu_times) - sum(start_cpu_times)

        print(f"load CPU时间: {cpu_time:.3f} 核秒")
        print(f"load CPU使用率: {end_cpu - start_cpu}%")
        print(f"load 内存使用: {(end_mem - start_mem) / (1024 * 1024):.2f} MB")
        print(f"load 运行时间: {end_time - start_time:.2f} 秒")

        return result

    return wrapper


# 计算空余度量标准
def calculate_free_space_metric(service_index, original_traffic, instances):
    traffic = original_traffic[:, service_index]
    instance_count = instances[:, service_index]
    # 避免除以0的错误，给实例数加一个小常数
    return traffic / (instance_count + 1e-6)


# 筛选出需要负载均衡的服务器（根据空余度量标准进行筛选）
def get_servers_to_balance(original_traffic, instances, ratio_threshold, num_servers, num_services):
    servers_to_balance = []
    for server in range(num_servers):
        for service in range(num_services):
            ratio = original_traffic[server, service] / (instances[server, service] + 1e-6)
            if ratio > ratio_threshold:
                servers_to_balance.append((server, service))
                break
    return servers_to_balance


# 筛选出最有空余的服务器（根据空余度量标准进行排序）
def get_most_free_space_servers(service_index, original_traffic, instances, edge_only=True):
    free_space_metric = calculate_free_space_metric(service_index, original_traffic, instances)
    if edge_only:
        free_space_metric = free_space_metric[:-1]  # 排除云服务器
    return np.argsort(-free_space_metric)  # 降序排序


# 计算可以分配的最大流量
def calculate_max_transferable_traffic(target_server, service,
                                       original_traffic, instances, ratio_threshold):
    # 计算要使目标服务器不超过阈值的最大可分配流量
    current_traffic = original_traffic[target_server, service]
    current_instances = instances[target_server, service]
    max_allowable_traffic = ratio_threshold * (current_instances + 1e-6)
    max_transferable_traffic = max_allowable_traffic - current_traffic
    return max_transferable_traffic


# 重分配流量
def redistribute_traffic(original_traffic, instances, ratio_threshold):
    num_servers = len(instances)
    num_services = len(instances[0])
    servers_to_balance = get_servers_to_balance(original_traffic, instances, ratio_threshold, num_servers, num_services)
    for server, service in servers_to_balance:
        if original_traffic[server, service] > 0:
            # 优先分配给边缘服务器
            most_free_space_servers = get_most_free_space_servers(service, original_traffic,
                                                                  instances, edge_only=True)
            redistributed = False
            for target_server in most_free_space_servers:
                if target_server != server:
                    # 计算可转移的最大流量
                    max_transferable_traffic = calculate_max_transferable_traffic(target_server, service,
                                                                                  original_traffic, instances,
                                                                                  ratio_threshold)
                    if max_transferable_traffic > 0:
                        # 实际转移的流量不超过可转移的最大流量
                        traffic_to_transfer = min(original_traffic[server, service], max_transferable_traffic, 10)
                        original_traffic[server, service] -= traffic_to_transfer
                        original_traffic[target_server, service] += traffic_to_transfer
                        # 检查是否还需要负载均衡
                        ratio = original_traffic[server, service] / (instances[server, service] + 1e-6)
                        if ratio <= ratio_threshold:
                            redistributed = True
                            break
            if not redistributed:
                # 如果边缘服务器无法满足要求，再分配给云服务器
                cloud_server = num_servers - 1
                max_transferable_traffic = calculate_max_transferable_traffic(cloud_server, service,
                                                                              original_traffic, instances,
                                                                              ratio_threshold)
                if max_transferable_traffic > 0:
                    traffic_to_transfer = min(original_traffic[server, service], max_transferable_traffic, 10)
                    original_traffic[server, service] -= traffic_to_transfer
                    original_traffic[cloud_server, service] += traffic_to_transfer
    return original_traffic
