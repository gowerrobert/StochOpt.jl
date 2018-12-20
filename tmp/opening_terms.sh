#!/bin/sh

gnome-terminal -- sh -c "julia ./tmp/script_args.jl australian; exec bash"
