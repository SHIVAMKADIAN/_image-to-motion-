#!/bin/bash
# Image Motion Studio — Start JavaScript (Node.js) Server
set -e

cd "$(dirname "$0")"
node backend/server.js
