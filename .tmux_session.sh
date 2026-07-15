#!/bin/bash
SESSION=$1

tmux send-keys -t "$SESSION:1" ". .venv/bin/activate" Enter
tmux send-keys -t "$SESSION:1" 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))")' Enter
tmux send-keys -t "$SESSION:1" "nvim" Enter
