BEGIN;

CREATE SCHEMA IF NOT EXISTS gemini_gateway;

TRUNCATE TABLE
    gemini_gateway.route_attempts,
    gemini_gateway.cooldowns,
    gemini_gateway.quota_windows,
    gemini_gateway.model_limits,
    gemini_gateway.key_proxy_bindings,
    gemini_gateway.api_keys,
    gemini_gateway.proxy_endpoints,
    gemini_gateway.google_projects
RESTART IDENTITY CASCADE;

INSERT INTO gemini_gateway.google_projects
SELECT * FROM soybob_v3.google_projects;

INSERT INTO gemini_gateway.proxy_endpoints
SELECT * FROM soybob_v3.proxy_endpoints;

INSERT INTO gemini_gateway.api_keys
SELECT * FROM soybob_v3.api_keys;

INSERT INTO gemini_gateway.key_proxy_bindings
SELECT * FROM soybob_v3.key_proxy_bindings;

INSERT INTO gemini_gateway.model_limits
SELECT * FROM soybob_v3.model_limits;

INSERT INTO gemini_gateway.quota_windows
SELECT * FROM soybob_v3.quota_windows;

INSERT INTO gemini_gateway.cooldowns
SELECT * FROM soybob_v3.cooldowns;

INSERT INTO gemini_gateway.route_attempts
SELECT * FROM soybob_v3.route_attempts;

SELECT setval(pg_get_serial_sequence('gemini_gateway.google_projects', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.google_projects), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.proxy_endpoints', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.proxy_endpoints), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.api_keys', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.api_keys), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.key_proxy_bindings', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.key_proxy_bindings), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.model_limits', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.model_limits), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.quota_windows', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.quota_windows), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.cooldowns', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.cooldowns), 1), true);
SELECT setval(pg_get_serial_sequence('gemini_gateway.route_attempts', 'id'), COALESCE((SELECT max(id) FROM gemini_gateway.route_attempts), 1), true);

COMMIT;
