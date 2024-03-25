#!/bin/bash
# 
# 写一个可以在mac下运行的shell脚本，包含如下几个功能:
# 1. 查找文件夹及子文件中文件大于45MB的文件
# 2. 将大于45MB的文件新建一个目录,在这个目录中将之切割成40MB的文件,并在这个目录中新建一个一键合并的shell脚本
# 3. 在生成拆分文件后,删除原始超过45MB的文件.并且打印这些文件名字和路径
# 4. 在切分文件的文件夹中，生成一个脚本可以一键合并生成拆分的文件，确保这些文件在原路径
# 

# 定义存储分割文件的目录
split_dir="split_files"
# 创建存储分割文件的目录
mkdir -p "$split_dir"
# 初始化合并脚本
merge_script="$split_dir/merge_files.sh"
echo "#!/bin/bash" > "$merge_script"

# 查找大于45MB的文件，并排除.git目录
find . -type f -size +45M ! -path '*/.git/*' -print0 | while IFS= read -r -d '' file; do
  # 打印文件名和路径
  echo "找到文件: $file"
  
  # 获取文件名
  filename=$(basename "${file}")
  # 定义分割后文件存放目录
  output_directory=$(dirname "${file}")
  # 分割文件，每个文件40MB大小
  split -b 40M "$file" "$split_dir/$filename.part"
  
  # 删除原文件
  rm "$file"

  # 在merge脚本中写入合并这些分割文件的命令
  # 将多个分割后的文件合并，并将合并后的文件移动回原始路径
  echo "cat \"$split_dir/$filename.part\"* > \"$output_directory/$filename\"" >> "$merge_script"
  # 合并后删除分割文件
  echo "rm \"$split_dir/$filename.part\"*" >> "$merge_script"
done

# 赋予合并脚本执行权限
chmod +x "$merge_script"

# 提示用户
echo "分割完成，原始文件已删除。合并脚本位于 '$merge_script'。"

