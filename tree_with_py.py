import os

def export_py_tree(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 깊이에 따른 들여쓰기
            depth = dirpath.replace(root_dir, "").count(os.sep)
            indent = "    " * depth
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")

            sub_indent = "    " * (depth + 1)
            for filename in filenames:
                if filename.endswith(".py"):
                    f.write(f"{sub_indent}{filename}\n")

if __name__ == "__main__":
    current_dir = os.getcwd()  # 현재 디렉토리
    output_path = os.path.join(current_dir, "py_tree.txt")

    export_py_tree(current_dir, output_path)
    print(f".py 파일 트리가 '{output_path}'에 저장되었습니다.")