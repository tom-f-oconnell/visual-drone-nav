#!/usr/bin/env bash

. venv/bin/activate
cd ground
uvicorn planner.app:app --port 8347 --host 0.0.0.0
