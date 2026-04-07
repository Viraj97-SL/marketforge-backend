-- MarketForge AI — PostgreSQL initialisation script
-- Run once on a fresh Railway PostgreSQL instance.
-- docker-compose maps this to /docker-entrypoint-initdb.d/01_init.sql

-- Create airflow database (Airflow requires its own database)
CREATE DATABASE airflow;

-- Enable required extensions (PostgreSQL 16)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create the market schema (all platform tables live here)
CREATE SCHEMA IF NOT EXISTS market;

-- Grant schema access to the application role
GRANT USAGE ON SCHEMA market TO marketforge;
GRANT CREATE ON SCHEMA market TO marketforge;

-- Airflow needs its own schema (separate from market)
CREATE SCHEMA IF NOT EXISTS airflow;
GRANT ALL PRIVILEGES ON SCHEMA airflow TO marketforge;

-- Note: actual table DDL is managed by init_database() in memory/postgres.py
-- using CREATE TABLE IF NOT EXISTS, making it idempotent.
