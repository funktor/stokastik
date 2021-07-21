import sys
import psycopg2
import logging
import rds_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    conn = psycopg2.connect(database=rds_config.db_name, 
                            user=rds_config.db_username, 
                            password=rds_config.db_password, 
                            host=rds_config.db_host, 
                            port="5432")
    
except Exception as e:
    logger.error("ERROR: Unexpected error: Could not connect to RDS instance.")
    logger.error(e)
    sys.exit()


def handler(event, context):
    logger.info("Creating schema, table and indexes...")
    
    with conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS hstore;')
        cur.execute('CREATE SCHEMA IF NOT EXISTS user_graph;')
        cur.execute('CREATE TABLE IF NOT EXISTS user_graph.inverted_index(id text primary key, attribute_name_value text, profile_id text);')
        cur.execute('CREATE TABLE IF NOT EXISTS user_graph.adjacency_list(id text primary key, src_profile_id text, dst_profile_id text);')
        cur.execute('CREATE INDEX ON user_graph.inverted_index((id));')
        cur.execute('CREATE INDEX ON user_graph.inverted_index((attribute_name_value));')
        cur.execute('CREATE INDEX ON user_graph.inverted_index((profile_id));')
        cur.execute('CREATE INDEX ON user_graph.adjacency_list((id));')
        cur.execute('CREATE INDEX ON user_graph.adjacency_list((src_profile_id));')
        cur.execute('TRUNCATE user_graph.inverted_index;')
        cur.execute('TRUNCATE user_graph.adjacency_list;')
        cur.execute('DROP FUNCTION IF EXISTS get_profiles(source_id varchar);')
        cur.execute('''
                    CREATE OR REPLACE FUNCTION get_profiles(source_id text) RETURNS setof text AS
                    $func$
                    DECLARE
                        visited_ids hstore;
                        queue text[];
                        x text;
                        r user_graph.adjacency_list%rowtype;
                    BEGIN
                        visited_ids := concat('"', source_id, '"=>"1"')::hstore;
                        queue := queue || source_id;

                        WHILE array_length(queue, 1) > 0
                        LOOP
                            x := queue[1];
                            queue := queue[2:array_length(queue,1)];

                            FOR r in SELECT * FROM user_graph.adjacency_list WHERE src_profile_id=x
                            LOOP
                                IF NOT visited_ids ? r.dst_profile_id THEN
                                    queue := queue || r.dst_profile_id;
                                    visited_ids := visited_ids || concat('"', r.dst_profile_id, '"=>"1"')::hstore;
                                END IF;
                            END LOOP;
                        END LOOP;

                        RETURN QUERY SELECT skeys(visited_ids) as profiles;
                    END
                    $func$ LANGUAGE plpgsql;
                    ''')
    
    conn.commit()

    return "Created user_graph"