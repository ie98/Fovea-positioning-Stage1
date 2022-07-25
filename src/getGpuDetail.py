import pynvml


# 字节数转GB
def bytes_to_gb(sizes):
    sizes = round(sizes / (1024 ** 3), 2)
    return f"{sizes} GB"


def gpu(num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(num)  # 显卡句柄（只有一张显卡）
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # 显卡名称
    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显卡内存信息
    data = dict(
        gpu_name=gpu_name.decode("utf-8"),
        gpu_memory_total=bytes_to_gb(gpu_memory.total),
        gpu_memory_used=bytes_to_gb(gpu_memory.used),
        gpu_memory_free=bytes_to_gb(gpu_memory.free),
    )
    return data


if __name__=='__main__':
    gpu_0 = gpu(0)
    gpu_1 = gpu(1)
    print(gpu_0)
    print(gpu_1)
    free = []
    free.append(float(gpu_0['gpu_memory_free'].split(' ')[0]))
    free.append(float(gpu_1['gpu_memory_free'].split(' ')[0]))
    print(free)

