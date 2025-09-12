#!/bin/bash

# USAGE
# dest is the only positional argument and is the data root directory to transfer subdirectories into (should end with a /,
# include user@host: at the start for remote transfers). rsync needs this root directory to already exist.
# -r flag will allow for remote transfers using ssh private key
send_to_remote=false
while getopts "r:" opt; do
    case "$opt" in
        r)
            send_to_remote=true
            private_key_path="$OPTARG"
            ;;
        *)
            echo "Invalid argumnet"
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1)) # move past options to positional arg

dest=$1
if [[ -z "$dest" ]]; then
    echo "Must specify a destination to send to"
    exit 1
fi

echo "Copying data/ to destination $dest"

IFS=" "
readarray -t data_subdirs <<< "$(cd $HOME/acai-omr/data; ls -d */)"
for subdir in "${data_subdirs[@]}"; do
    echo "Starting $subdir transfer"
    if [[ "$send_to_remote" == true ]]; then
        echo "Sending to remote using private key at $private_key_path"
        rsync -az --progress -e "ssh -i $private_key_path" "$HOME/acai-omr/data/$subdir" "$dest$subdir" &
    else
        rsync -a --progress "$HOME/acai-omr/data/$subdir" "$dest$subdir" &
    fi
done

wait
echo "Done transferring"