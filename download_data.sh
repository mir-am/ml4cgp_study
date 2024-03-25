#!/bin/bash

# Check if a path argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

# The user-provided path
accepted_path="$1"

# Create the target directory if it doesn't already exist
target_directory="${accepted_path}/ml4cgp_study_data/"
mkdir -p "$target_directory"

# DOI URL pointing to the "latest version" of the record
doi_url="https://zenodo.org/doi/10.5281/zenodo.10638852"

# Use curl to follow the redirect and get the final URL, extracting the record ID
record_id=$(curl -Ls -o /dev/null -w %{url_effective} "$doi_url" | grep -oP 'records/\K\d+')

echo "Record: $record_id"

# Temporary file for concatenated archive
temp_concat_file="${target_directory}ml4cgp_study_data.tar.lz4"

# Fetch and iterate through the file entries, downloading each matching file and preparing for extraction
curl -s "https://zenodo.org/api/records/$record_id" | jq -r '.files[] | select(.key | test("\\.lz4\\.[0-9]{2}$")) | .links.self + " " + .key' | while IFS=' ' read -r url filename; do
    local_file="${target_directory}${filename}"
    echo "Downloading $filename to $target_directory ..."
    curl -o "$local_file" "$url"
    echo "Downloaded $filename"
    
    # Concatenate parts if they are part of a multi-part archive
    if [[ $filename =~ \.lz4\.[0-9]{2}$ ]]; then
        cat "$local_file" >> "$temp_concat_file"
        echo "Added $filename to the archive."
    fi
done

# Extract the concatenated .lz4 file
echo "Extracting $temp_concat_file ..."
lz4 -d "$temp_concat_file" "${temp_concat_file%.lz4}"
echo "Extraction complete."

# Remove the temporary concatenated .lz4 file after extraction
rm "$temp_concat_file"
echo "Cleanup complete."
