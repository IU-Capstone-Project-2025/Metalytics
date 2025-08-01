\connect metalytics;

CREATE TABLE IF NOT EXISTS metals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS historical_prices (
    id SERIAL PRIMARY KEY,
    metal_id INTEGER NOT NULL REFERENCES metals(id),
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 6) NOT NULL
);

CREATE TABLE IF NOT EXISTS predicted_prices (
    id SERIAL PRIMARY KEY,
    metal_id INTEGER NOT NULL REFERENCES metals(id),
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 6) NOT NULL
);

CREATE TABLE IF NOT EXISTS predicted_prices_II (
    id SERIAL PRIMARY KEY,
    metal_id INTEGER NOT NULL REFERENCES metals(id),
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 6) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_historical_prices_metal_timestamp ON historical_prices (metal_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_metal_timestamp ON predicted_prices (metal_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_metal_timestamp ON predicted_prices_II (metal_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_prices_timestamp ON historical_prices (timestamp);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_timestamp ON predicted_prices (timestamp);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_timestamp ON predicted_prices_II (timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_prices_metal ON historical_prices (metal_id);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_metal ON predicted_prices (metal_id);
CREATE INDEX IF NOT EXISTS idx_predicted_prices_metal ON predicted_prices_II (metal_id);

INSERT INTO metals (id, name) VALUES 
    (1, 'gold'),
    (2, 'silver'),
    (3, 'platinum');