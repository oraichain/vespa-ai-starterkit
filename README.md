# Vespa Search Application

## Overview

This project involves creating a search application using Vespa, a platform for scalable and fast data serving. The main objective is to process movie data, deploy a Vespa instance in Docker, and execute various types of searches. The tasks include data processing, application deployment, and query execution.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Steps to Complete the Assignment](#steps-to-complete-the-assignment)
   - [Data Processing](#1-data-processing)
   - [Run Vespa as a Docker Container](#2-run-vespa-as-a-docker-container)
   - [Configure Vespa and Ingest Data](#3-configure-vespa-and-ingest-data)
   - [Run Search Queries](#4-run-search-queries)

## Prerequisites

- **Python 3.x**
- **Docker Desktop** (Ensure it is installed and running)
- **vespacli** or **pyvespa** Python module

## Steps to Complete the Assignment

### 1. Data Processing

1. **Run the provided script** to process `tmdb_5000_movies.csv` into a Vespa-compatible JSON format.
   ```python
   from process_script import process_tmdb_csv
   process_tmdb_csv("tmdb_5000_movies.csv", "clean_tmdb.jsonl")
   ```
2. **Verify the output**: Ensure that `clean_tmdb.jsonl` contains the required fields (`doc_id`, `title`, and `text`).

### 2. Run Vespa as a Docker Container

1. **Pull and Run Vespa Container**:
   ```bash
   docker compose up -d
   ```
2. **Verify the Container**:
   - Run `docker ps` to confirm the container is running.
   - Access `http://localhost:19071` to check the deployment API.

### 3. Configure Vespa and Ingest Data

1. **Install `vespacli`**:
   ```bash
   pip install --ignore-installed vespacli
   ```
2. **Deploy the Application**:
   ```bash
   vespa config set target local
   vespa deploy --wait 300 app
   ```
3. **Feed Data into Vespa**:
   ```bash
   vespa feed -t http://localhost:8080 clean_tmdb.jsonl
   ```

### 4. Run Search Queries

1. **Connect to Vespa Using Python**:

   ```python
   from vespa.application import Vespa

   app = Vespa(url="http://localhost", port=8080)
   ```

2. **Run Keyword Search**:
   ```python
   df = keyword_search(app, "Harry Potter and the Half-Blood Prince")
   print(df)
   ```
3. **Run Semantic Search**:
   ```python
   df = semantic_search(app, "Harry Potter and the Half-Blood Prince")
   print(df)
   ```
