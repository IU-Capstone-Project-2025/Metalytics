CREATE DATABASE metalytics;

\connect metalytics;

CREATE TABLE IF NOT EXISTS historical_prices (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    price DECIMAL(20, 6) NOT NULL
);

CREATE TABLE IF NOT EXISTS predicted_prices (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    price DECIMAL(20, 6) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_historical_prices_timestamp ON historical_prices (timestamp);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_timestamp ON predicted_prices (timestamp);