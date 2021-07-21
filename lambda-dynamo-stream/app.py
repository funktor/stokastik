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
import hashlib

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
    
def rds_inverted_index_remove(profile_id):
    logger.info('Removing inverted index entry...');
    with conn.cursor() as cur:
        cur.execute('DELETE FROM user_graph.inverted_index WHERE profile_id=%s', (profile_id,))
    conn.commit()
    
def rds_inverted_index_insert_update(profile_id, metadata):
    logger.info('Updating inverted index entry...');
    with conn.cursor() as cur:
        for k, v in metadata.items():
            s = k + '::' + v
            h = s + '||' + str(profile_id)
            uid = str(hashlib.md5(h.encode()).hexdigest())
        
            cur.execute('INSERT INTO user_graph.inverted_index(id, attribute_name_value, profile_id) VALUES (%s, %s, %s) ON CONFLICT(id) DO UPDATE SET attribute_name_value=%s, profile_id=%s', (uid, s, profile_id, s, profile_id))
            
    conn.commit()
    
def rds_adjacency_list_remove(profile_id):
    logger.info('Removing adjacency list entry...');
    with conn.cursor() as cur:
        cur.execute('DELETE FROM user_graph.adjacency_list WHERE src_profile_id=%s OR dst_profile_id=%s', (profile_id, profile_id))
    conn.commit()
    
def rds_adjacency_list_insert_update(profile_id, metadata):
    logger.info('Updating adjacency list entry...');
    profiles = set()
    
    with conn.cursor() as cur:
        for k, v in metadata.items():
            s = k + '::' + v
            cur.execute('SELECT profile_id FROM user_graph.inverted_index WHERE attribute_name_value=%s', (s, ))
            rows = cur.fetchall()
            
            for row in rows:
                p = row[0]
                
                if p != profile_id:
                    profiles.add(p)
    conn.commit()
    
    with conn.cursor() as cur:
        for p in profiles:
            s1 = profile_id + '||' + p
            s2 = p + '||' + profile_id

            uid1 = str(hashlib.md5(s1.encode()).hexdigest())
            uid2 = str(hashlib.md5(s2.encode()).hexdigest())

            cur.execute('INSERT INTO user_graph.adjacency_list(id, src_profile_id, dst_profile_id) VALUES (%s, %s, %s) ON CONFLICT(id) DO UPDATE SET src_profile_id=%s, dst_profile_id=%s', (uid1, profile_id, p, profile_id, p))

            cur.execute('INSERT INTO user_graph.adjacency_list(id, src_profile_id, dst_profile_id) VALUES (%s, %s, %s) ON CONFLICT(id) DO UPDATE SET src_profile_id=%s, dst_profile_id=%s', (uid2, p, profile_id, p, profile_id))
    
    conn.commit()


def handler(event, context):
    for record in event['Records']:
        op = record['eventName']
        profile_id = str(record['dynamodb']['Keys']['profile_id']['S'])
        metadata = json.loads(record['dynamodb']['NewImage']['metadata']['S'])
        
        if op == 'REMOVE' or op == 'DELETE' or op == 'UPDATE' or op == 'MODIFY':
            rds_inverted_index_remove(profile_id)
            rds_adjacency_list_remove(profile_id)
            
        if op == 'INSERT' or op == 'UPDATE' or op == 'MODIFY':
            rds_inverted_index_insert_update(profile_id, metadata)
            rds_adjacency_list_insert_update(profile_id, metadata)
        
        inv_idx, adj_lst = [], []
        
        with conn.cursor() as cur:
            cur.execute('SELECT * from user_graph.inverted_index')
            rows = cur.fetchall()
            
            for row in rows:
                inv_idx.append(row)
        
        logger.info(profile_id + ' : ' + json.dumps(inv_idx))
        
        with conn.cursor() as cur:
            cur.execute('SELECT * from user_graph.adjacency_list')
            rows = cur.fetchall()
            
            for row in rows:
                adj_lst.append(row)
        
        logger.info(profile_id + ' : ' + json.dumps(adj_lst))
        
        logger.info('Doing Union Find...')
        
        with conn.cursor() as cur:
            cur.execute('SELECT get_profiles(%s) as profile', (profile_id,))
            rows = cur.fetchall()
            
            for row in rows:
                logger.info(profile_id + ' : ' + json.dumps(row[0]))

    return "Updated RDS"