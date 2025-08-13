import os

def export_directory_tree(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            depth = dirpath.replace(root_dir, "").count(os.sep)
            indent = "    " * depth
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")

            sub_indent = "    " * (depth + 1)
            for filename in filenames:
                f.write(f"{sub_indent}{filename}\n")

if __name__ == "__main__":
    current_dir = os.getcwd()  # 현재 디렉토리
    output_path = os.path.join(current_dir, "directory_tree.txt")

    export_directory_tree(current_dir, output_path)
    print(f"디렉토리 트리 구조가 '{output_path}'에 저장되었습니다.")
