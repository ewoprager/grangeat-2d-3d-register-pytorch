#!/usr/bin/env bash
set -e

# Allow Qt X11 forwarding
export QT_X11_NO_MITSHM=1

exec python -m reg23_app