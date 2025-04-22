# See https://github.com/openai/Video-Pre-Training#contractor-demonstrations for links to all data
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 split_file_path" >&2
    exit 1
fi
split_file_path=$1
basepath="https://openaipublic.blob.core.windows.net/minecraft-rl/data/6.13/"
suffixes=( "mp4" "jsonl" )
output_dir="minerl_data"
# loop through each line in the split file
while IFS= read -r line; do
    # get name without suffix from line
    name=$(echo "$line" | cut -f 1 -d '.')
    echo $name
    for suffix in "${suffixes[@]}"; do
        # download file
        wget -P "$output_dir" "$basepath$name.$suffix"
    done
done < "$split_file_path"