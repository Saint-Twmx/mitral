def find_longest_consecutive_subsequence(numbers, step=1):
    max_len = 0
    current_len = 1
    start_index = 0  # 追踪当前连续序列的起始索引

    # 用于存储最长序列的起始和结束索引
    best_start = 0
    best_end = 0

    for i in range(1, len(numbers)):
        if numbers[i] <= numbers[i - 1] + step:
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = start_index
                best_end = i - 1
            start_index = i
            current_len = 1

    # 检查最后一个序列是否是最长的
    if current_len > max_len:
        max_len = current_len
        best_start = start_index
        best_end = len(numbers) - 1

    return numbers[best_start:best_end + 1]


# # 测试数据
# numbers = [1, 2, 3, 5, 6, 8, 9, 15, 16, 17, 18, 19, 12]
# longest_subsequence = find_longest_consecutive_subsequence(numbers, step=3)
# print("最长连续子序列:", longest_subsequence)
