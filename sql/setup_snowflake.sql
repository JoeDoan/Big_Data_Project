-- Create database and schema
CREATE DATABASE IF NOT EXISTS CS5542_DB;
USE DATABASE CS5542_DB;

CREATE SCHEMA IF NOT EXISTS APP;
USE SCHEMA APP;

-- Create table to store chat history
CREATE TABLE IF NOT EXISTS chat_history (
    session_id VARCHAR(100),
    timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    role VARCHAR(20),
    content TEXT
);
