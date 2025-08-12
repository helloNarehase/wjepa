import torch
import random

def create_fixed_span_masks(
    seq_length,
    num_targets=4,
    target_span_len=8,
    context_scale_range=(0.85, 1.0),
    allow_target_overlap=True
):
    """
    I-JEPA 스타일 마스크 생성 (Text 버전, 고정 길이 Span, 인덱스 반환)

    Args:
        seq_length (int): 전체 토큰 길이 (L)
        num_targets (int): 타겟 블록 개수 (M)
        target_span_len (int): 각 타겟 블록의 고정 길이 (Span)
        context_scale_range (tuple): 컨텍스트 블록 길이 비율 범위
        allow_target_overlap (bool): 타겟 블록 간 겹침 허용 여부

    Returns:
        tuple[list[list[int]], list[int]]:
            - target_indices_list: 각 타겟 블록의 인덱스 리스트 (M개)
            - context_indices: 컨텍스트 블록의 인덱스 리스트
    """
    target_indices_list = []
    all_target_indices = set()

    # 1. Target 블록 인덱스 생성
    for _ in range(num_targets):
        if allow_target_overlap:
            # 겹침 허용 시: 단순히 랜덤 시작점 선택
            max_start_idx = seq_length - target_span_len
            if max_start_idx < 0:
                print("Warning: target_span_len is greater than seq_length.")
                break
            start_idx = random.randint(0, max_start_idx)
        else:
            # 겹침 비허용 시: 겹치지 않는 연속 공간 탐색
            available_indices = sorted(list(set(range(seq_length)) - all_target_indices))
            if len(available_indices) < target_span_len:
                break  # 남은 공간이 타겟 길이보다 작으면 중단

            valid_starts = []
            # 'available_indices'에서 연속된 'target_span_len' 길이의 공간이 있는지 확인
            for i in range(len(available_indices) - target_span_len + 1):
                potential_start = available_indices[i]
                potential_end = available_indices[i + target_span_len - 1]
                if potential_end - potential_start == target_span_len - 1:
                    valid_starts.append(potential_start)

            if not valid_starts:
                break  # 가능한 시작점이 없으면 중단
            start_idx = random.choice(valid_starts)

        block_indices = list(range(start_idx, start_idx + target_span_len))
        target_indices_list.append(block_indices)
        all_target_indices.update(block_indices)

    # 2. Context 블록 인덱스 생성
    context_len = random.randint(
        int(seq_length * context_scale_range[0]),
        int(seq_length * context_scale_range[1])
    )
    context_start = random.randint(0, seq_length - context_len)
    context_indices = set(range(context_start, context_start + context_len))

    # 타겟과 겹치는 부분은 컨텍스트에서 제외
    context_indices -= all_target_indices

    return target_indices_list, sorted(list(context_indices))


# ===== 실행 예제 =====
if __name__ == "__main__":
    seq_len = 100
    num_targets = 4
    span_len = 10  # 모든 타겟 마스크의 길이를 10으로 고정

    print("--- 겹침 허용 (allow_target_overlap=True) ---")
    targets_overlap, context_overlap = create_fixed_span_masks(
        seq_len,
        num_targets=num_targets,
        target_span_len=span_len,
        allow_target_overlap=True
    )

    print(f"고정 Target Span 길이: {span_len}")
    print("Target Indices:")
    for i, idx_list in enumerate(targets_overlap):
        print(f"  Block {i+1} (길이={len(idx_list)}): {idx_list}")
    print("Context Indices:", context_overlap)

    print("\n" + "="*40 + "\n")

    print("--- 겹침 비허용 (allow_target_overlap=False) ---")
    targets_no_overlap, context_no_overlap = create_fixed_span_masks(
        seq_len,
        num_targets=num_targets,
        target_span_len=span_len,
        allow_target_overlap=False
    )

    print(f"고정 Target Span 길이: {span_len}")
    print("Target Indices:")
    for i, idx_list in enumerate(targets_no_overlap):
        print(f"  Block {i+1} (길이={len(idx_list)}): {idx_list}")
    print("Context Indices:", context_no_overlap)