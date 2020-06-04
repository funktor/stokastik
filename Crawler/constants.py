WIKI_SEED_URL='https://en.wikipedia.org/wiki/Cache-oblivious_algorithm'
ELASTICACHE_URL='redis-cluster.7icodg.clustercfg.usw2.cache.amazonaws.com'
ELASTICACHE_PORT=6379
ELASTICACHE_QUEUE_KEY='task_queue'
ELASTICACHE_QUEUE_KEY_AMZN='amz_task_queue'
BLOOM_FILTER_SIZE=10000121
BLOOM_FILTER_NUM_HASHES=5
AMZN_URL_SET='amazon_url_set'
AWS_KEYSPACES_PEM='/home/ec2-user/AmazonRootCA1.pem'
CASSANDRA_URL='cassandra.us-west-2.amazonaws.com'
CASSANDRA_PORT=9142
WIKI_KEYSPACE_NAME='wiki_crawler'
AMZN_KEYSPACE_NAME='amzn_crawler'
NUM_THREADS=100
WIKI_MAX_LEVEL=3
WIKI_MAX_URLS_PER_PAGE=10
THROTTLE_TIME=1.0
REDIS_BLOCKING_TIMEOUT=5

WIKI_CREATE_TABLE_SQL='CREATE TABLE IF NOT EXISTS crawler(url_hash text PRIMARY KEY, url text, url_text text, parent_url_hash text, inserted_time timestamp);'
WIKI_INSERT_PREP_STMT='INSERT INTO crawler(url, url_hash, url_text, parent_url_hash, inserted_time) VALUES (?, ?, ?, ?, ?)'

AMZN_CREATE_TABLE_SQL_SEARCH='CREATE TABLE IF NOT EXISTS amzn_search(url_hash text, url text, query text, search_result_index uuid, metadata text, inserted_time timestamp, PRIMARY KEY((url_hash, search_result_index)));'
AMZN_INSERT_PREP_STMT_SEARCH='INSERT INTO amzn_search(url, url_hash, query, search_result_index, metadata, inserted_time) VALUES (?, ?, ?, ?, ?, ?)'

AMZN_CREATE_TABLE_SQL_DETAILS='CREATE TABLE IF NOT EXISTS amzn_details(url_hash text PRIMARY KEY, url text, query text, metadata text, inserted_time timestamp);'
AMZN_INSERT_PREP_STMT_DETAILS='INSERT INTO amzn_details(url, url_hash, query, metadata, inserted_time) VALUES (?, ?, ?, ?, ?)'

WIKI_OUT_FILE='wiki_crawl.csv'
AMZN_OUT_FILE='amzn_crawl.csv'
LOGGER='crawler.log'

CLUSTER_MODE=True
