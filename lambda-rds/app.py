import random
import uuid, json, string
from random import randrange
from datetime import timedelta, datetime
import collections, time, json
import sys
import os
import psycopg2
import math
import socket
import struct
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
    
    
def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def handler(event, context):
    logger.info("Generating data...")
    
    n, m = 10**3, 10**4
    
    ip_addresses = set()

    for i in range(n):
        addr = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
        ip_addresses.add(addr)

    ip_addresses = list(ip_addresses)

    d1 = datetime.strptime('01/01/2021 1:30 PM', '%m/%d/%Y %I:%M %p')
    d2 = datetime.strptime('06/30/2021 4:50 AM', '%m/%d/%Y %I:%M %p')

    dates = []
    for i in range(m):
        dates.append(random_date(d1, d2))

    dates = [str(x) for x in sorted(dates)]

    ips = random.choices(ip_addresses, k=m)

    data = []
    for i in range(m):
        data.append((i+1, ips[i], dates[i]))
    
    logger.info("Creating schema, table and indexes...")
    
    with conn.cursor() as cur:
        cur.execute('CREATE SCHEMA IF NOT EXISTS cardinality;')
        cur.execute('CREATE TABLE IF NOT EXISTS cardinality.ip_addresses(id integer primary key, ip_address varchar(20), created_at timestamp);')
        cur.execute('CREATE INDEX ON cardinality.ip_addresses((ip_address));')
        cur.execute('CREATE INDEX ON cardinality.ip_addresses((created_at));')
        cur.execute('TRUNCATE cardinality.ip_addresses;')
    
    conn.commit()
    
    logger.info("Inserting data...")
    
    batch_size = 10**3
    num_batches = int(math.ceil(len(data)/batch_size))
    
    for i in range(num_batches):
        logger.info("Insertion started for batch " + str(i))

        start, end = i*batch_size, min(len(data), (i+1)*batch_size)
        
        with conn.cursor() as cur:
            args_str = ','.join(cur.mogrify("(%s,%s,TIMESTAMP %s)", x).decode('utf-8') for x in data[start:end])
            cur.execute('INSERT INTO cardinality.ip_addresses(id, ip_address, created_at) VALUES ' + args_str)
            
        conn.commit()
        
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT(ip_address)) as count FROM cardinality.ip_addresses")
        row = cur.fetchone()
        
        if row is not None and len(row) > 0:
            logger.info("Unique data count = " + str(row[0]))

    return "Added %d items from RDS MySQL table" %(m)